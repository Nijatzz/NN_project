import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class MixedEmitter(nn.Module):
    # z_t -> emission parameters for three distribution families
    # x layout: first n_studentt cols  = Student-t features
    #           next  n_lognormal cols  = log-normal features
    #           remaining n_normal cols = Gaussian features
    def __init__(self, z_dim, hidden_dim, n_studentt, n_lognormal, n_normal):
        super().__init__()
        self.n_studentt  = n_studentt
        self.n_lognormal = n_lognormal
        self.n_normal    = n_normal
        self.shared_st = nn.Linear(z_dim, hidden_dim)
        self.shared_ln = nn.Linear(z_dim, hidden_dim)
        self.shared_n  = nn.Linear(z_dim, hidden_dim)
        self.st_mu    = nn.Linear(hidden_dim, n_studentt)
        self.st_sigma = nn.Linear(hidden_dim, n_studentt)
        self.log_df   = nn.Linear(hidden_dim, n_studentt)
        self.ln_mu    = nn.Linear(hidden_dim, n_lognormal)
        self.ln_sigma = nn.Linear(hidden_dim, n_lognormal)
        self.n_mu     = nn.Linear(hidden_dim, n_normal)
        self.n_sigma  = nn.Linear(hidden_dim, n_normal)
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        h_st = torch.tanh(self.shared_st(z_t))
        h_ln = torch.tanh(self.shared_ln(z_t))
        h_n  = torch.tanh(self.shared_n(z_t))
        df       = 2.0 + self.softplus(self.log_df(h_st))
        st_mu    = self.st_mu(h_st)
        st_sigma = self.softplus(self.st_sigma(h_st))
        ln_mu    = self.ln_mu(h_ln)
        ln_sigma = self.softplus(self.ln_sigma(h_ln))
        n_mu     = self.n_mu(h_n)
        n_sigma  = self.softplus(self.n_sigma(h_n))
        return df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma


class GatedTransition(nn.Module):
    # maps z_{t-1} -> parameters of p(z_t | z_{t-1})
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.lin_z_gate               = nn.Linear(z_dim, hidden_dim)
        self.lin_z_gate_reverse       = nn.Linear(hidden_dim, z_dim)
        self.lin_z_transition         = nn.Linear(z_dim, hidden_dim)
        self.lin_z_transition_reverse = nn.Linear(hidden_dim, z_dim)
        self.lin_z_mu                 = nn.Linear(z_dim, z_dim)
        self.lin_z_sigma              = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.lin_z_mu.weight)
        nn.init.zeros_(self.lin_z_mu.bias)
        self.act      = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, z_prev):
        g_t = self.act(self.lin_z_gate(z_prev))
        g_t = self.sigmoid(self.lin_z_gate_reverse(g_t))
        h_t = self.act(self.lin_z_transition(z_prev))
        h_t = self.lin_z_transition_reverse(h_t)
        mu_t    = (1 - g_t) * self.lin_z_mu(z_prev) + g_t * h_t
        sigma_t = self.softplus(self.lin_z_sigma(self.act(h_t)))
        return mu_t, sigma_t


class ConditionedGatedTransition(nn.Module):
    # maps (z_{t-1}, cond) -> parameters of p(z_t | z_{t-1}, cond)
    # cond is fixed for a block of steps; concatenated with z_prev as joint input
    def __init__(self, z_dim, cond_dim, hidden_dim):
        super().__init__()
        in_dim = z_dim + cond_dim
        self.lin_gate               = nn.Linear(in_dim, hidden_dim)
        self.lin_gate_reverse       = nn.Linear(hidden_dim, z_dim)
        self.lin_transition         = nn.Linear(in_dim, hidden_dim)
        self.lin_transition_reverse = nn.Linear(hidden_dim, z_dim)
        self.lin_z_mu               = nn.Linear(z_dim, z_dim)   # identity baseline on z only
        self.lin_z_sigma            = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.lin_z_mu.weight)
        nn.init.zeros_(self.lin_z_mu.bias)
        self.act      = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, z_prev, cond):
        inp = torch.cat([z_prev, cond], dim=-1)
        g_t = self.act(self.lin_gate(inp))
        g_t = self.sigmoid(self.lin_gate_reverse(g_t))
        h_t = self.act(self.lin_transition(inp))
        h_t = self.lin_transition_reverse(h_t)
        mu_t    = (1 - g_t) * self.lin_z_mu(z_prev) + g_t * h_t
        sigma_t = self.softplus(self.lin_z_sigma(self.act(h_t)))
        return mu_t, sigma_t


class DoublyConditionedGatedTransition(nn.Module):
    # maps (f_{t-1}, sc_tau, sm_tau) -> parameters of p(f_t | f_{t-1}, sc_tau, sm_tau)
    # both slow states are fixed for K steps; all three concatenated as joint input
    def __init__(self, f_dim, sc_dim, sm_dim, hidden_dim):
        super().__init__()
        in_dim = f_dim + sc_dim + sm_dim
        self.lin_gate               = nn.Linear(in_dim, hidden_dim)
        self.lin_gate_reverse       = nn.Linear(hidden_dim, f_dim)
        self.lin_transition         = nn.Linear(in_dim, hidden_dim)
        self.lin_transition_reverse = nn.Linear(hidden_dim, f_dim)
        self.lin_z_mu               = nn.Linear(f_dim, f_dim)   # identity baseline on f only
        self.lin_z_sigma            = nn.Linear(f_dim, f_dim)
        nn.init.eye_(self.lin_z_mu.weight)
        nn.init.zeros_(self.lin_z_mu.bias)
        self.act      = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, f_prev, sc_tau, sm_tau):
        inp = torch.cat([f_prev, sc_tau, sm_tau], dim=-1)
        g_t = self.act(self.lin_gate(inp))
        g_t = self.sigmoid(self.lin_gate_reverse(g_t))
        h_t = self.act(self.lin_transition(inp))
        h_t = self.lin_transition_reverse(h_t)
        mu_t    = (1 - g_t) * self.lin_z_mu(f_prev) + g_t * h_t
        sigma_t = self.softplus(self.lin_z_sigma(self.act(h_t)))
        return mu_t, sigma_t


class Combiner(nn.Module):
    # maps (z_{t-1}, h_right_t) -> parameters of q(z_t | z_{t-1}, x_{1:T})
    def __init__(self, z_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_mu    = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_sigma = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_z     = nn.Linear(z_dim, rnn_hidden_dim)
        self.act      = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_prev, h_right):
        h_combined = 0.5 * (self.act(self.lin_z(z_prev)) + h_right)
        mu_t    = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class CorporateCombiner(nn.Module):
    # maps (sc_{t-1}, sm_{t-1}, h_right_t) -> parameters of q(sc_t | sc_{t-1}, sm_{t-1}, x_{1:T})
    # three-way blend: previous corporate state, previous macro state, RNN context
    def __init__(self, sc_dim, sm_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_sc    = nn.Linear(sc_dim, rnn_hidden_dim)
        self.lin_sm    = nn.Linear(sm_dim, rnn_hidden_dim)
        self.lin_mu    = nn.Linear(rnn_hidden_dim, sc_dim)
        self.lin_sigma = nn.Linear(rnn_hidden_dim, sc_dim)
        self.act      = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, sc_prev, sm_prev, h_right):
        h_combined = (self.act(self.lin_sc(sc_prev)) + self.act(self.lin_sm(sm_prev)) + h_right) / 3.0
        mu_t    = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class FastCombiner(nn.Module):
    # maps (f_{t-1}, sc_tau, sm_tau, h_right_t) -> parameters of q(f_t | f_{t-1}, sc_tau, sm_tau, x_{1:T})
    # four-way blend: previous fast state, current corporate state, current macro state, RNN context
    def __init__(self, f_dim, sc_dim, sm_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_f     = nn.Linear(f_dim, rnn_hidden_dim)
        self.lin_sc    = nn.Linear(sc_dim, rnn_hidden_dim)
        self.lin_sm    = nn.Linear(sm_dim, rnn_hidden_dim)
        self.lin_mu    = nn.Linear(rnn_hidden_dim, f_dim)
        self.lin_sigma = nn.Linear(rnn_hidden_dim, f_dim)
        self.act      = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, f_prev, sc_tau, sm_tau, h_right, sm_prev=None, sc_prev=None):
        h_combined = (self.act(self.lin_f(f_prev)) + self.act(self.lin_sc(sc_tau)) +
                      self.act(self.lin_sm(sm_tau)) + h_right) / 4.0
        mu_t    = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class FiLMFastCombiner(nn.Module):
    # FiLM-conditioned variant of FastCombiner.
    # h_base     = 0.5 * (act(lin_f(f_prev)) + h_right)        [original DMM base]
    # film_input = cat([sm_prev, sm_tau, sc_prev, sc_tau])
    # gamma, beta = film_layer(film_input).chunk(2)
    # h_combined = gamma * h_base + beta
    def __init__(self, f_dim, sc_dim, sm_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_f      = nn.Linear(f_dim, rnn_hidden_dim)
        self.film_layer = nn.Linear(2 * sm_dim + 2 * sc_dim, 2 * rnn_hidden_dim)
        self.lin_mu     = nn.Linear(rnn_hidden_dim, f_dim)
        self.lin_sigma  = nn.Linear(rnn_hidden_dim, f_dim)
        self.act        = nn.Tanh()
        self.softplus   = nn.Softplus()

    def forward(self, f_prev, sc_tau, sm_tau, h_right, sm_prev, sc_prev):
        h_base     = 0.5 * (self.act(self.lin_f(f_prev)) + h_right)
        film_input = torch.cat([sm_prev, sm_tau, sc_prev, sc_tau], dim=-1)
        gamma, beta = self.film_layer(film_input).chunk(2, dim=-1)
        h_combined = gamma * h_base + beta
        mu_t    = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class TripleDMM(nn.Module):
    def __init__(self,
                 n_st_fast, n_ln_fast, n_n_fast,
                 n_st_corp, n_ln_corp, n_n_corp,
                 n_st_macro, n_ln_macro, n_n_macro,
                 sm_dim, sc_dim, f_dim,
                 hidden_dim,
                 rnn_fast_hidden, rnn_corp_hidden, rnn_macro_hidden,
                 rnn_num_layers,
                 K,
                 combiner='naive'):
        super().__init__()
        self.n_st_fast  = n_st_fast
        self.n_ln_fast  = n_ln_fast
        self.n_n_fast   = n_n_fast
        self.n_st_corp  = n_st_corp
        self.n_ln_corp  = n_ln_corp
        self.n_n_corp   = n_n_corp
        self.n_st_macro = n_st_macro
        self.n_ln_macro = n_ln_macro
        self.n_n_macro  = n_n_macro
        self.sm_dim     = sm_dim
        self.sc_dim     = sc_dim
        self.f_dim      = f_dim
        self.K          = K

        n_fast  = n_st_fast  + n_ln_fast  + n_n_fast
        n_corp  = n_st_corp  + n_ln_corp  + n_n_corp
        n_macro = n_st_macro + n_ln_macro + n_n_macro

        self.macro_emitter    = MixedEmitter(sm_dim, hidden_dim, n_st_macro, n_ln_macro, n_n_macro)
        self.corp_emitter     = MixedEmitter(sc_dim, hidden_dim, n_st_corp,  n_ln_corp,  n_n_corp)
        self.fast_emitter     = MixedEmitter(f_dim,  hidden_dim, n_st_fast,  n_ln_fast,  n_n_fast)

        self.macro_transition = GatedTransition(sm_dim, hidden_dim)
        self.corp_transition  = ConditionedGatedTransition(sc_dim, sm_dim, hidden_dim)
        self.fast_transition  = DoublyConditionedGatedTransition(f_dim, sc_dim, sm_dim, hidden_dim)

        self.macro_combiner   = Combiner(sm_dim, rnn_macro_hidden)
        self.corp_combiner    = CorporateCombiner(sc_dim, sm_dim, rnn_corp_hidden)
        if combiner == 'film':
            self.fast_combiner = FiLMFastCombiner(f_dim, sc_dim, sm_dim, rnn_fast_hidden)
        else:
            self.fast_combiner = FastCombiner(f_dim, sc_dim, sm_dim, rnn_fast_hidden)

        self.macro_rnn        = nn.GRU(n_macro, rnn_macro_hidden, rnn_num_layers, batch_first=True)
        self.corp_rnn         = nn.GRU(n_corp,  rnn_corp_hidden,  rnn_num_layers, batch_first=True)
        self.fast_rnn         = nn.GRU(n_fast,  rnn_fast_hidden,  rnn_num_layers, batch_first=True)

        self.sm_0       = nn.Parameter(torch.zeros(sm_dim))
        self.sc_0       = nn.Parameter(torch.zeros(sc_dim))
        self.f_0        = nn.Parameter(torch.zeros(f_dim))
        self.h_macro_0  = nn.Parameter(torch.zeros(rnn_num_layers, 1, rnn_macro_hidden))
        self.h_corp_0   = nn.Parameter(torch.zeros(rnn_num_layers, 1, rnn_corp_hidden))
        self.h_fast_0   = nn.Parameter(torch.zeros(rnn_num_layers, 1, rnn_fast_hidden))

    def _split_fast(self, x):
        a = self.n_st_fast
        b = self.n_st_fast + self.n_ln_fast
        return x[..., :a], x[..., a:b], x[..., b:]

    def _split_corp(self, x):
        a = self.n_st_corp
        b = self.n_st_corp + self.n_ln_corp
        return x[..., :a], x[..., a:b], x[..., b:]

    def _split_macro(self, x):
        a = self.n_st_macro
        b = self.n_st_macro + self.n_ln_macro
        return x[..., :a], x[..., a:b], x[..., b:]

    def _emit_obs(self, emitter, z, x_st, x_ln, x_n, t, prefix):
        df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma = emitter(z)
        if emitter.n_studentt > 0:
            pyro.sample(f"{prefix}_st_{t}",
                        dist.StudentT(df, st_mu, st_sigma).to_event(1),
                        obs=x_st)
        if emitter.n_lognormal > 0:
            pyro.sample(f"{prefix}_ln_{t}",
                        dist.LogNormal(ln_mu, ln_sigma).to_event(1),
                        obs=x_ln)
        if emitter.n_normal > 0:
            pyro.sample(f"{prefix}_n_{t}",
                        dist.Normal(n_mu, n_sigma).to_event(1),
                        obs=x_n)

    def model(self, x_fast, x_corp, x_macro):
        # x_fast:  (batch, T_fast,  n_fast)   daily
        # x_corp:  (batch, T_slow,  n_corp)   monthly
        # x_macro: (batch, T_slow,  n_macro)  monthly  T_slow = T_fast // K
        pyro.module("triple_dmm", self)
        batch_size, T_fast, _ = x_fast.shape
        T_slow = x_macro.shape[1]

        x_fast_st,  x_fast_ln,  x_fast_n  = self._split_fast(x_fast)
        x_corp_st,  x_corp_ln,  x_corp_n  = self._split_corp(x_corp)
        x_macro_st, x_macro_ln, x_macro_n = self._split_macro(x_macro)

        sm_prev = self.sm_0.expand(batch_size, self.sm_dim)
        sc_prev = self.sc_0.expand(batch_size, self.sc_dim)
        f_prev  = self.f_0.expand(batch_size, self.f_dim)

        with pyro.plate("data", batch_size):
            for tau in range(T_slow):
                # 1. Macro state — pure Markov on sm_prev
                sm_mean, sm_std = self.macro_transition(sm_prev)
                sm_tau = pyro.sample(f"sm_{tau}",
                                     dist.Normal(sm_mean, sm_std).to_event(1))
                self._emit_obs(self.macro_emitter, sm_tau,
                               x_macro_st[:, tau, :],
                               x_macro_ln[:, tau, :],
                               x_macro_n[:, tau, :],
                               tau, "x_macro")

                # 2. Corporate state — conditioned on sc_prev and sm_prev (previous macro)
                sc_mean, sc_std = self.corp_transition(sc_prev, sm_prev)
                sc_tau = pyro.sample(f"sc_{tau}",
                                     dist.Normal(sc_mean, sc_std).to_event(1))
                self._emit_obs(self.corp_emitter, sc_tau,
                               x_corp_st[:, tau, :],
                               x_corp_ln[:, tau, :],
                               x_corp_n[:, tau, :],
                               tau, "x_corp")

                # 3. Fast states — conditioned on current sm_tau and sc_tau
                for k in range(self.K):
                    t = tau * self.K + k
                    if t >= T_fast:
                        break
                    f_mean, f_std = self.fast_transition(f_prev, sc_tau, sm_tau)
                    f_t = pyro.sample(f"f_{t}",
                                      dist.Normal(f_mean, f_std).to_event(1))
                    self._emit_obs(self.fast_emitter, f_t,
                                   x_fast_st[:, t, :],
                                   x_fast_ln[:, t, :],
                                   x_fast_n[:, t, :],
                                   t, "x_fast")
                    f_prev = f_t

                sm_prev = sm_tau
                sc_prev = sc_tau

    def guide(self, x_fast, x_corp, x_macro):
        # Strict t-1 posterior: RNN context for step tau/t is built from x[0:tau]/x[0:t] only.
        # Achieved by feeding x[:-1] to each RNN and prepending h_0 for step 0.
        # h_macro_ctx[:, tau, :] = context from x_macro[0 : tau]  (excludes x_macro[tau])
        # h_corp_ctx[:,  tau, :] = context from x_corp[0 : tau]   (excludes x_corp[tau])
        # h_fast_ctx[:,  t,   :] = context from x_fast[0 : t]     (excludes x_fast[t])
        pyro.module("triple_dmm", self)
        batch_size, T_fast, _ = x_fast.shape
        T_slow = x_macro.shape[1]

        h_macro_0 = self.h_macro_0.expand(-1, batch_size, -1).contiguous()
        h_macro_enc, _ = self.macro_rnn(x_macro[:, :-1, :], h_macro_0)
        h_macro_ctx = torch.cat(
            [h_macro_0[-1].unsqueeze(1), h_macro_enc], dim=1
        )  # (batch, T_slow, rnn_macro_hidden)

        h_corp_0 = self.h_corp_0.expand(-1, batch_size, -1).contiguous()
        h_corp_enc, _ = self.corp_rnn(x_corp[:, :-1, :], h_corp_0)
        h_corp_ctx = torch.cat(
            [h_corp_0[-1].unsqueeze(1), h_corp_enc], dim=1
        )  # (batch, T_slow, rnn_corp_hidden)

        h_fast_0 = self.h_fast_0.expand(-1, batch_size, -1).contiguous()
        h_fast_enc, _ = self.fast_rnn(x_fast[:, :-1, :], h_fast_0)
        h_fast_ctx = torch.cat(
            [h_fast_0[-1].unsqueeze(1), h_fast_enc], dim=1
        )  # (batch, T_fast, rnn_fast_hidden)

        sm_prev = self.sm_0.expand(batch_size, self.sm_dim)
        sc_prev = self.sc_0.expand(batch_size, self.sc_dim)
        f_prev  = self.f_0.expand(batch_size, self.f_dim)

        with pyro.plate("data", batch_size):
            for tau in range(T_slow):
                # Macro posterior
                sm_mean, sm_std = self.macro_combiner(sm_prev, h_macro_ctx[:, tau, :])
                sm_tau = pyro.sample(f"sm_{tau}",
                                     dist.Normal(sm_mean, sm_std).to_event(1))

                # Corporate posterior — uses sm_prev to match model dependency p(sc|sc_prev, sm_prev)
                sc_mean, sc_std = self.corp_combiner(sc_prev, sm_prev, h_corp_ctx[:, tau, :])
                sc_tau = pyro.sample(f"sc_{tau}",
                                     dist.Normal(sc_mean, sc_std).to_event(1))

                # Fast posterior — uses current sm_tau and sc_tau
                for k in range(self.K):
                    t = tau * self.K + k
                    if t >= T_fast:
                        break
                    f_mean, f_std = self.fast_combiner(f_prev, sc_tau, sm_tau, h_fast_ctx[:, t, :], sm_prev, sc_prev)
                    f_t = pyro.sample(f"f_{t}",
                                      dist.Normal(f_mean, f_std).to_event(1))
                    f_prev = f_t

                sm_prev = sm_tau
                sc_prev = sc_tau

    def infer(self, x_fast, x_corp, x_macro):
        # Strict t-1 inference: mirrors guide() exactly but returns means/stds.
        # h_macro_ctx[:, tau, :] = context from x_macro[0 : tau]  (excludes x_macro[tau])
        # h_corp_ctx[:,  tau, :] = context from x_corp[0 : tau]   (excludes x_corp[tau])
        # h_fast_ctx[:,  t,   :] = context from x_fast[0 : t]     (excludes x_fast[t])
        #
        # Returns:
        #   sm_means, sm_stds: (batch, T_slow, sm_dim)
        #   sc_means, sc_stds: (batch, T_slow, sc_dim)
        #   f_means,  f_stds:  (batch, T_fast, f_dim)
        batch_size, T_fast, _ = x_fast.shape
        T_slow = x_macro.shape[1]

        h_macro_0 = self.h_macro_0.expand(-1, batch_size, -1).contiguous()
        h_macro_enc, _ = self.macro_rnn(x_macro[:, :-1, :], h_macro_0)
        h_macro_ctx = torch.cat(
            [h_macro_0[-1].unsqueeze(1), h_macro_enc], dim=1
        )  # (batch, T_slow, rnn_macro_hidden)

        h_corp_0 = self.h_corp_0.expand(-1, batch_size, -1).contiguous()
        h_corp_enc, _ = self.corp_rnn(x_corp[:, :-1, :], h_corp_0)
        h_corp_ctx = torch.cat(
            [h_corp_0[-1].unsqueeze(1), h_corp_enc], dim=1
        )  # (batch, T_slow, rnn_corp_hidden)

        h_fast_0 = self.h_fast_0.expand(-1, batch_size, -1).contiguous()
        h_fast_enc, _ = self.fast_rnn(x_fast[:, :-1, :], h_fast_0)
        h_fast_ctx = torch.cat(
            [h_fast_0[-1].unsqueeze(1), h_fast_enc], dim=1
        )  # (batch, T_fast, rnn_fast_hidden)

        sm_prev = self.sm_0.expand(batch_size, self.sm_dim)
        sc_prev = self.sc_0.expand(batch_size, self.sc_dim)
        f_prev  = self.f_0.expand(batch_size, self.f_dim)

        sm_means, sm_stds = [], []
        sc_means, sc_stds = [], []
        f_means,  f_stds  = [], []

        for tau in range(T_slow):
            sm_mean, sm_std = self.macro_combiner(sm_prev, h_macro_ctx[:, tau, :])
            sm_means.append(sm_mean)
            sm_stds.append(sm_std)
            sm_tau = sm_mean

            sc_mean, sc_std = self.corp_combiner(sc_prev, sm_prev, h_corp_ctx[:, tau, :])
            sc_means.append(sc_mean)
            sc_stds.append(sc_std)
            sc_tau = sc_mean

            for k in range(self.K):
                t = tau * self.K + k
                if t >= T_fast:
                    break
                f_mean, f_std = self.fast_combiner(f_prev, sc_tau, sm_tau, h_fast_ctx[:, t, :], sm_prev, sc_prev)
                f_means.append(f_mean)
                f_stds.append(f_std)
                f_prev = f_mean

            sm_prev = sm_mean
            sc_prev = sc_mean

        sm_means = torch.stack(sm_means, dim=1)   # (batch, T_slow, sm_dim)
        sm_stds  = torch.stack(sm_stds,  dim=1)
        sc_means = torch.stack(sc_means, dim=1)   # (batch, T_slow, sc_dim)
        sc_stds  = torch.stack(sc_stds,  dim=1)
        f_means  = torch.stack(f_means,  dim=1)   # (batch, T_fast, f_dim)
        f_stds   = torch.stack(f_stds,   dim=1)

        # Emission parameters from posterior-mean latent states.
        # Each emitter is applied to the full time dimension at once.
        # Returns (df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma),
        # each shaped (batch, T, n_*); empty last dim when that family has 0 features.
        def _emit_params(emitter, z_means, T):
            B = z_means.shape[0]
            raw = emitter(z_means.reshape(B * T, -1))
            return tuple(o.reshape(B, T, -1) for o in raw)

        macro_emit = _emit_params(self.macro_emitter, sm_means, T_slow)
        corp_emit  = _emit_params(self.corp_emitter,  sc_means, T_slow)
        fast_emit  = _emit_params(self.fast_emitter,  f_means,  T_fast)

        # macro_emit / corp_emit / fast_emit:
        #   [0] df        (batch, T, n_studentt)
        #   [1] st_mu     (batch, T, n_studentt)
        #   [2] st_sigma  (batch, T, n_studentt)
        #   [3] ln_mu     (batch, T, n_lognormal)
        #   [4] ln_sigma  (batch, T, n_lognormal)
        #   [5] n_mu      (batch, T, n_normal)
        #   [6] n_sigma   (batch, T, n_normal)

        return (sm_means, sm_stds, sc_means, sc_stds, f_means, f_stds,
                macro_emit, corp_emit, fast_emit)
