from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

class TC_VAE:
    def __init__(self, seq_len: int = 60, random_state: int = 42):
        self.seq_len = seq_len

        self.model = None
        self.data = None

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def train(self,
              data: pd.DataFrame,
              epochs: int = 500,
              batch_size: int = 128,
              lr: float = 0.001):
        self.data = data.copy()
        prices = data["open"].to_numpy(dtype=np.float32)

        raw_windows = make_windows(prices, seq_len=self.seq_len, stride=1)
        windows = raw_windows / raw_windows[:, :1, :]  # each starts at 1.0

        cfg = default_config(seq_len=self.seq_len, data_dim=1, conditional=False,
                             transform="log", model="BetaVAE")
        cfg.latent_dim = 2
        cfg.E_hidden_dim = 16
        cfg.D_hidden_dim = 16
        cfg.E_num_layers = 2
        cfg.D_num_layers = 2
        cfg.P_num_flows = 3
        cfg.P_hidden_dim = 250
        cfg.beta = 0.04
        cfg.lr = lr
        cfg.batch_size = batch_size
        cfg.epochs = epochs

        self.model = TCVAE(cfg)
        train_tcvae(self.model, windows, epochs=cfg.epochs, verbose=True)

    def generate(self):
        syn_long = generate_long_paths(self.model,
                                       n_paths=1,
                                       length=len(self.data),
                                       p0=self.data['open'].iloc[0],
                                       seq_len=self.seq_len)[0]
        return pd.DataFrame({
            'timestamp': self.data['timestamp'].values,
            'open': syn_long
        })


# ===========================================================================
#  init_weights
# ===========================================================================
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data,
                                gain=nn.init.calculate_gain("relu"))
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass


# ===========================================================================
#  Reconstruction losses
# ===========================================================================
def l1_loss(x: Tensor, recon_x: Tensor) -> Tensor:
    return torch.abs(recon_x - x).mean(dim=0).sum()

def l2_loss(x: Tensor, recon_x: Tensor) -> Tensor:
    return torch.square(recon_x - x).mean(dim=0).sum()

_LOSSES = {"l1": l1_loss, "l2": l2_loss}

# ===========================================================================
#  Transforms
# ===========================================================================
class _Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x

class _Log(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x.log()

class _Exp(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x.exp()

def get_transform(name: str) -> nn.Module:
    if name in ("", "id"):  return _Identity()
    if name == "log":       return _Log()
    if name == "exp":       return _Exp()
    raise ValueError(f"Unknown transform: {name!r}")


# ===========================================================================
#  Gaussian helpers
# ===========================================================================
_PI = torch.tensor(math.pi)

def log_standard_normal(x: Tensor) -> Tensor:
    log_p = -0.5 * torch.log(2.0 * _PI.to(x.device)) - 0.5 * x ** 2
    return log_p.sum(dim=-1)

def entropy_normal(log_var: Tensor) -> Tensor:
    entropy = -0.5 * (1.0 + torch.log(2.0 * _PI.to(log_var.device)) + log_var)
    return entropy.sum(dim=-1)


# ===========================================================================
#  Priors
# ===========================================================================
class BasePrior(nn.Module):
    def sample(self, n_sample: int, device) -> Tensor: raise NotImplementedError
    def log_prob(self, x: Tensor) -> Tensor:          raise NotImplementedError


class GaussianPrior(BasePrior):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def sample(self, n_sample: int, device) -> Tensor:
        return torch.randn(n_sample, self.dim, device=device)

    def log_prob(self, x: Tensor) -> Tensor:
        return log_standard_normal(x)


class FlowPrior(BasePrior):
    def __init__(self, num_flows: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        L, M = latent_dim, hidden_dim
        assert L % 2 == 0, "FlowPrior requires an even flattened latent dim"

        # ``nets`` ends in Tanh (bounded scale -- numerical stability)
        def nets():
            return nn.Sequential(
                nn.Linear(L // 2, M), nn.LeakyReLU(),
                nn.Linear(M, M),      nn.LeakyReLU(),
                nn.Linear(M, L // 2), nn.Tanh(),
            )

        def nett():
            return nn.Sequential(
                nn.Linear(L // 2, M), nn.LeakyReLU(),
                nn.Linear(M, M),      nn.LeakyReLU(),
                nn.Linear(M, L // 2),
            )

        self.D = latent_dim
        self.num_flows = num_flows
        self.s = nn.ModuleList([nets() for _ in range(num_flows)])
        self.t = nn.ModuleList([nett() for _ in range(num_flows)])

    # ------------------------------------------------------------------
    def coupling(self, x: Tensor, index: int, forward: bool):
        xa, xb = torch.chunk(x, 2, dim=1)
        s = self.s[index](xa)
        t = self.t[index](xa)
        if forward:
            yb = (xb - t) * torch.exp(-s)
        else:
            yb = torch.exp(s) * xb + t
        return torch.cat((xa, yb), dim=1), s

    @staticmethod
    def permute(x: Tensor) -> Tensor:
        return x.flip(1)

    def f(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_J = x.new_zeros(x.shape[0])
        z = x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)
        return z, log_det_J

    def f_inv(self, z: Tensor) -> Tensor:
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)
        return x

    # ------------------------------------------------------------------
    def sample(self, n_sample: int, device) -> Tensor:
        z = torch.randn(n_sample, self.D, device=device)
        return self.f_inv(z)

    def log_prob(self, x: Tensor) -> Tensor:
        z, log_det_J = self.f(x)
        return log_standard_normal(z) + log_det_J


# ===========================================================================
#  Conditioner
# ===========================================================================
class IdentityConditioner(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Identity()
    def forward(self, c: Tensor) -> Tensor: return self.net(c)


# ===========================================================================
#  Encoders / Decoders
# ===========================================================================
class LSTMEncoder(nn.Module):
    def __init__(self, data_dim: int, data_length: int,
                 latent_dim: int, latent_length: int,
                 hidden_dim: int, n_layers: int):
        super().__init__()
        self.data_dim     = data_dim
        self.data_length  = data_length
        self.latent_dim   = latent_dim
        self.latent_length = latent_length
        self.hidden_dim   = hidden_dim
        self.activation   = nn.Tanh()

        self.linear_pre  = nn.Linear(data_dim,   hidden_dim)
        self.lstm        = nn.LSTM(input_size=hidden_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=n_layers,
                                   batch_first=True)
        self.linear_post = nn.Linear(hidden_dim, hidden_dim)

        self.mean_net    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), self.activation,
            nn.Linear(hidden_dim, latent_dim),
        )
        self.log_var_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), self.activation,
            nn.Linear(hidden_dim, latent_dim),
        )

        self.lstm.apply(init_weights)
        self.linear_pre.apply(init_weights)
        self.linear_post.apply(init_weights)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: (B, T, data_dim)
        x = F.relu(self.linear_pre(x))
        x, _ = self.lstm(x)
        x = self.linear_post(self.activation(x))
        mean    = self.mean_net(x)                        # (B, T, latent_dim)
        log_var = self.log_var_net(x)
        # flatten time into the latent dim: (B, T * latent_dim)
        return mean.flatten(start_dim=1), log_var.flatten(start_dim=1)


class LSTMResEncoder(LSTMEncoder):
    def __init__(self, data_dim, data_length, latent_dim, latent_length,
                 hidden_dim, n_layers):
        super().__init__(data_dim, data_length, latent_dim, latent_length,
                         hidden_dim, n_layers)
        # Re-bind linear_post with the residual input width
        self.linear_post0 = nn.Linear(hidden_dim + data_dim, hidden_dim)
        self.linear_post  = nn.Linear(hidden_dim + hidden_dim + data_dim,
                                      hidden_dim)
        self.linear_post0.apply(init_weights)
        self.linear_post.apply(init_weights)

    def forward(self, x0: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.linear_pre(x0))
        x, _ = self.lstm(x)
        x = self.activation(x)                            # LSTM out, tanh'd

        y0 = torch.cat([x, x0], dim=-1)                   # residual w/ input
        y  = self.activation(self.linear_post0(y0))

        z0 = torch.cat([y, y0], dim=-1)                   # 2nd residual
        z  = self.activation(self.linear_post(z0))

        mean    = self.mean_net(z)
        log_var = self.log_var_net(z)
        return mean.flatten(start_dim=1), log_var.flatten(start_dim=1)


class CLSTMResEncoder(LSTMResEncoder):
    def __init__(self, data_dim, data_length, latent_dim, latent_length,
                 hidden_dim, n_layers, condition_dim, conditioner):
        # NOTE: the super expects the FULL input width (= data_dim + cond_dim)
        super().__init__(data_dim + condition_dim, data_length,
                         latent_dim, latent_length, hidden_dim, n_layers)
        self.data_dim_real = data_dim
        self.condition_dim = condition_dim
        self.conditioner   = conditioner

    def forward(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.data_length, 1)
        return super().forward(torch.cat([x, c], dim=-1))


class LSTMDecoder(nn.Module):
    def __init__(self, data_dim, data_length, latent_dim, latent_length,
                 hidden_dim, n_layers, activation=nn.Tanh()):
        super().__init__()
        self.data_dim     = data_dim
        self.data_length  = data_length
        self.latent_dim   = latent_dim
        self.latent_length = latent_length
        self.activation   = activation

        self.linear_pre  = nn.Linear(latent_dim, hidden_dim)
        self.lstm        = nn.LSTM(input_size=hidden_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=n_layers,
                                   batch_first=True)
        self.linear_post = nn.Linear(hidden_dim, data_dim)

        self.lstm.apply(init_weights)
        self.linear_post.apply(init_weights)
        self.linear_pre.apply(init_weights)

    def forward(self, noise: Tensor) -> Tensor:
        noise = noise.view(-1, self.latent_length, self.latent_dim)
        x = self.activation(self.linear_pre(noise))
        x, _ = self.lstm(x)
        return self.linear_post(self.activation(x))


class LSTMResDecoder(LSTMDecoder):
    def __init__(self, data_dim, data_length, latent_dim, latent_length,
                 hidden_dim, n_layers, activation=nn.Tanh()):
        super().__init__(data_dim, data_length, latent_dim, latent_length,
                         hidden_dim, n_layers, activation)
        self.linear_post0 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.linear_post  = nn.Linear(hidden_dim, data_dim)
        self.linear_post0.apply(init_weights)
        self.linear_post.apply(init_weights)

    def forward(self, noise: Tensor) -> Tensor:
        noise = noise.view(-1, self.latent_length, self.latent_dim)
        x = self.activation(self.linear_pre(noise))
        x, _ = self.lstm(x)
        y = torch.cat([x, noise], dim=-1)                 # residual w/ latent
        y = self.linear_post0(self.activation(y))
        y = self.linear_post(self.activation(y))
        return y


class CLSTMResDecoder(LSTMResDecoder):
    def __init__(self, data_dim, data_length, latent_dim, latent_length,
                 hidden_dim, n_layers, condition_dim, conditioner,
                 activation=nn.ReLU()):
        super().__init__(data_dim, data_length, latent_dim, latent_length,
                         hidden_dim, n_layers, activation)
        self.condition_dim = condition_dim
        self.conditioner   = conditioner
        self.latent_dim_raw = latent_dim - condition_dim   # raw z width

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        c = self.conditioner(c)
        c = c.view(-1, 1, self.condition_dim).repeat(1, self.latent_length, 1)
        z = z.view(-1, self.latent_length, self.latent_dim_raw)
        return super().forward(torch.cat([z, c], dim=-1))


# ===========================================================================
#  MMD^2
# ===========================================================================
class _RBFKernel(nn.Module):
    def __init__(self, n_kernels: int = 5, mul_factor: float = 2.0,
                 bandwidth: Optional[float] = None):
        super().__init__()
        self.register_buffer("mults",
            mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))
        self.bandwidth = bandwidth

    def _band(self, L2):
        if self.bandwidth is not None: return self.bandwidth
        n = L2.shape[0]
        return L2.data.sum() / (n * n - n)

    def forward(self, X: Tensor) -> Tensor:
        L2 = torch.cdist(X, X) ** 2
        b  = self._band(L2)
        scaled = (b * self.mults)[:, None, None]
        return torch.exp(-L2[None, ...] / scaled).sum(dim=0)


class GaussianMMD2(nn.Module):
    def __init__(self, n_kernels: int = 5, mul_factor: float = 2.0,
                 bandwidth: Optional[float] = None):
        super().__init__()
        self.kernel = _RBFKernel(n_kernels, mul_factor, bandwidth)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        X = X.flatten(start_dim=1)
        Y = Y.flatten(start_dim=1)
        K = self.kernel(torch.vstack([X, Y]))
        n = X.shape[0]
        return K[:n, :n].mean() - 2 * K[:n, n:].mean() + K[n:, n:].mean()


# ===========================================================================
#  Configs and TCVAE
# ===========================================================================
@dataclass
class TCVAEConfig:
    # data_analysis / latent
    data_dim:       int = 1
    data_length:    int = 60
    latent_dim:     int = 2          # per-step latent
    latent_length:  int = 60         # latent has same length as data_analysis (causal)
    condition_dim:  int = 1          # set to 0 for unconditional

    # nets
    encoder:        str = "CLSTMRes"   # or "LSTMRes"
    decoder:        str = "CLSTMRes"   # or "LSTMRes"
    E_hidden_dim:   int = 16
    E_num_layers:   int = 2
    D_hidden_dim:   int = 16
    D_num_layers:   int = 2

    # prior
    prior:          str = "RealNVP"  # or "Gaussian"
    P_num_flows:    int = 3
    P_hidden_dim:   int = 250

    # transforms (e.g. price -> log -> model -> exp -> price)
    transform:      str = "log"
    inv_transform:  str = "exp"

    # loss
    model:          str = "InfoCVAE"    # or "BetaCVAE" / "BetaVAE"
    beta:           float = 0.04
    alpha:          float = 0.04
    reconstruction_loss: str = "l1"

    # training
    lr:             float = 1e-3
    epochs:         int = 500
    batch_size:     int = 256
    grad_clip:      float = 0.0
    device:         str = "cuda" if torch.cuda.is_available() else "cpu"

    # convenience flag: if False we ignore conditioner / labels
    conditional:    bool = True


def default_config(seq_len: int = 60, data_dim: int = 1,
                   conditional: bool = False, transform: str = "log",
                   model: str = "InfoCVAE", **kw) -> TCVAEConfig:
    cfg = TCVAEConfig(
        data_dim=data_dim, data_length=seq_len, latent_length=seq_len,
        transform=transform,
        inv_transform={"log": "exp", "id": "id", "": ""}.get(transform, "id"),
        condition_dim=1 if conditional else 0,
        encoder="CLSTMRes" if conditional else "LSTMRes",
        decoder="CLSTMRes" if conditional else "LSTMRes",
        conditional=conditional,
        model=model,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
class TCVAE(nn.Module):
    def __init__(self, cfg: TCVAEConfig):
        super().__init__()
        self.cfg = cfg

        self.conditioner = IdentityConditioner() if cfg.conditional else None

        if cfg.encoder == "CLSTMRes":
            assert cfg.conditional, "CLSTMRes encoder requires conditional=True"
            self.encoder = CLSTMResEncoder(
                cfg.data_dim, cfg.data_length, cfg.latent_dim,
                cfg.latent_length, cfg.E_hidden_dim, cfg.E_num_layers,
                cfg.condition_dim, self.conditioner)
        elif cfg.encoder == "LSTMRes":
            self.encoder = LSTMResEncoder(
                cfg.data_dim, cfg.data_length, cfg.latent_dim,
                cfg.latent_length, cfg.E_hidden_dim, cfg.E_num_layers)
        else:
            raise ValueError(f"Unknown encoder {cfg.encoder!r}")

        if cfg.decoder == "CLSTMRes":
            assert cfg.conditional
            self.decoder = CLSTMResDecoder(
                cfg.data_dim, cfg.data_length,
                cfg.latent_dim + cfg.condition_dim, cfg.latent_length,
                cfg.D_hidden_dim, cfg.D_num_layers,
                cfg.condition_dim, self.conditioner)
        elif cfg.decoder == "LSTMRes":
            self.decoder = LSTMResDecoder(
                cfg.data_dim, cfg.data_length, cfg.latent_dim,
                cfg.latent_length, cfg.D_hidden_dim, cfg.D_num_layers)
        else:
            raise ValueError(f"Unknown decoder {cfg.decoder!r}")

        flat_latent = cfg.latent_dim * cfg.latent_length
        if cfg.prior == "RealNVP":
            self.prior = FlowPrior(cfg.P_num_flows, flat_latent,
                                    cfg.P_hidden_dim)
        elif cfg.prior == "Gaussian":
            self.prior = GaussianPrior(flat_latent)
        else:
            raise ValueError(f"Unknown prior {cfg.prior!r}")

        self.recon_loss_fn = _LOSSES[cfg.reconstruction_loss]
        self.transform     = get_transform(cfg.transform)
        self.inv_transform = get_transform(cfg.inv_transform)
        self.mmd2          = GaussianMMD2() if cfg.model == "InfoCVAE" else None

        self.to(cfg.device)

    def _encode(self, x: Tensor, c: Optional[Tensor]):
        if self.cfg.conditional:
            return self.encoder(x, c)
        return self.encoder(x)

    def _decode(self, z: Tensor, c: Optional[Tensor]) -> Tensor:
        if self.cfg.conditional:
            return self.decoder(z, c)
        return self.decoder(z)

    @staticmethod
    def _sample_gauss(mu: Tensor, std: Tensor) -> Tensor:
        return mu + torch.randn_like(std) * std

    def forward(self, x0: Tensor, c: Optional[Tensor] = None) -> Dict:
        x = self.transform(x0)
        mu, log_var = self._encode(x, c)
        std = torch.exp(0.5 * log_var)
        z = self._sample_gauss(mu, std)               # (B, T*latent_dim)
        recon_x  = self._decode(z, c)
        recon_x0 = self.inv_transform(recon_x)

        recon_loss = self.recon_loss_fn(x0, recon_x0)
        # KL surrogate: -H(q) - E[log p(z)]
        post_term  = entropy_normal(log_var)            # (B,)
        prior_term = self.prior.log_prob(z)             # (B,)
        kl_loss    = (post_term - prior_term).mean()

        if self.cfg.model == "InfoCVAE":
            z_p = self.prior.sample(z.shape[0], device=z.device)
            mmd = self.mmd2(z, z_p)
            total = (recon_loss
                     + self.cfg.beta * kl_loss
                     + (self.cfg.alpha - self.cfg.beta) * mmd)
            return {"loss": total, "recon": recon_loss,
                    "kl": kl_loss, "mmd": mmd,
                    "recon_x": recon_x0, "z": z}

        if self.cfg.model in ("BetaCVAE", "BetaVAE"):
            total = recon_loss + self.cfg.beta * kl_loss
        elif self.cfg.model in ("VAE", "CVAE"):
            total = recon_loss + kl_loss
        else:
            raise ValueError(f"Unknown model {self.cfg.model!r}")
        return {"loss": total, "recon": recon_loss, "kl": kl_loss,
                "recon_x": recon_x0, "z": z}

    @torch.no_grad()
    def generate(self, n_sample: int,
                 c: Optional[Tensor] = None) -> np.ndarray:
        self.eval()
        device = self.cfg.device
        z = self.prior.sample(n_sample, device=device)
        if self.cfg.conditional:
            if c is None:
                c = torch.ones(n_sample, self.cfg.condition_dim, device=device)
            elif isinstance(c, np.ndarray):
                c = torch.tensor(c, dtype=torch.float32, device=device)
            else:
                c = c.to(device)
        recon_x  = self._decode(z, c)
        recon_x0 = self.inv_transform(recon_x)
        return recon_x0.cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x0, c=None) -> np.ndarray:
        self.eval()
        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float32)
        x0 = x0.to(self.cfg.device)
        if self.cfg.conditional and c is None:
            c = torch.ones(x0.shape[0], self.cfg.condition_dim,
                           device=self.cfg.device)
        elif isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32, device=self.cfg.device)
        out = self.forward(x0, c)
        return out["recon_x"].cpu().numpy()


def train_tcvae(model: TCVAE,
                data,
                labels=None,
                epochs: Optional[int] = None,
                batch_size: Optional[int] = None,
                lr: Optional[float] = None,
                verbose: bool = True) -> list:
    cfg = model.cfg
    epochs     = epochs     or cfg.epochs
    batch_size = batch_size or cfg.batch_size
    lr         = lr         or cfg.lr

    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    assert data.ndim == 3, "data_analysis must be (N, seq_len, data_dim)"
    assert data.shape[1] == cfg.data_length, (
        f"data_length mismatch: got {data.shape[1]}, "
        f"expected {cfg.data_length}")
    assert data.shape[2] == cfg.data_dim, (
        f"data_dim mismatch: got {data.shape[2]}, expected {cfg.data_dim}")

    if cfg.conditional:
        if labels is None:
            labels = torch.ones(data.shape[0], cfg.condition_dim)
        elif isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.float32)
        dataset = TensorDataset(data, labels)
    else:
        dataset = TensorDataset(data)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    model.train()
    for ep in range(1, epochs + 1):
        agg = {}
        n_batches = 0
        for batch in loader:
            if cfg.conditional:
                xb, cb = batch
                cb = cb.to(cfg.device)
            else:
                (xb,) = batch
                cb = None
            xb = xb.to(cfg.device)

            opt.zero_grad()
            out = model(xb, cb)
            out["loss"].backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                cfg.grad_clip)
            opt.step()

            for k, v in out.items():
                if isinstance(v, Tensor) and v.ndim == 0:
                    agg[k] = agg.get(k, 0.0) + v.item()
            n_batches += 1

        agg = {k: v / n_batches for k, v in agg.items()}
        history.append(agg)

        if verbose and (ep == 1 or ep % max(1, epochs // 20) == 0
                        or ep == epochs):
            msg = f"epoch {ep:4d}/{epochs} | " + " ".join(
                f"{k}={v:.4f}" for k, v in agg.items())
            print(msg)

    return history

def make_windows(series: np.ndarray, seq_len: int,
                 stride: int = 1) -> np.ndarray:
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]
    T = series.shape[0]
    n = (T - seq_len) // stride + 1
    return np.stack([series[i*stride : i*stride + seq_len] for i in range(n)])

def generate_long_paths(model, n_paths: int, length: int, p0: float,
                        seq_len: int) -> np.ndarray:
    # over-sample with a bit of safety margin
    n_windows_needed = (length + seq_len - 2) // (seq_len - 1) + 2
    raw = model.generate(n_paths * n_windows_needed)[..., 0]  # (M, 60)
    # re-normalise to start at 1.0 (the VAE output starts very near 1.0 but
    # not exactly, because of stochastic decoding)
    raw = raw / np.maximum(raw[:, :1], 1e-8)
    raw = raw.reshape(n_paths, n_windows_needed, seq_len)

    paths = np.empty((n_paths, length), dtype=np.float32)
    paths[:, 0] = p0
    cursor = 1
    for k in range(n_windows_needed):
        w = raw[:, k, :]  # (n_paths, 60)
        # multiplicative increments after the first day of the window
        inc = w[:, 1:] / w[:, :1]  # (n_paths, 59)
        end = min(cursor + inc.shape[1], length)
        n_take = end - cursor
        paths[:, cursor:end] = paths[:, cursor - 1:cursor] * inc[:, :n_take]
        cursor = end
        if cursor >= length:
            break
    return paths

