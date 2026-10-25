"""World model: learned dynamics + learned termination head + analytic observation/reward heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.world_model.dynamics import MLP, DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, 2) = [height error, tilt error] -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        self.dynamics = DynamicsMLP(hidden=cfg.hidden)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        feat = self.spec.canonical(s)
        delta = self.dynamics(feat, self.spec.mask_action(action), self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        return self.term_head(feats)

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n,70), term_bce (n,), logits (n,))."""
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat))
        logits = self.term_head(feats)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list[ReferenceFrame]):
        """k-step autoregressive loss.

        states: (k+1, n, 71) logged states; a_exec: (k, n, 29); xi: (n, 33);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1).
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w)
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
