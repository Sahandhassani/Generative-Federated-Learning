import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from fl_common import (
    evaluate_global,
    get_num_site_clients,
    create_models,
    get_combined_parameters,
    DEVICE,
    FL_DIR,
    OUTPUT_ROOT,
    train_local_client,
)
import os


class ClientWeightMLP(nn.Module):
    """
    Small neural network that maps per-client statistics to a scalar credit logit.
    It is intentionally simple because you only have 4 clients.
    Features:
      0 sample share
      1 validation loss
      2 validation SSIM
      3 validation PSNR
      4 round progress
      5 client id normalized
    """
    def __init__(self, in_dim=6, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FedLAWWeightModel(nn.Module):
    """
    Learnable aggregation logits for a fixed small number of clients.
    For your current setup this is suitable because the active hospital nodes
    are fixed by TOP_SITE_IDS and client_id is returned by the client.
    """
    def __init__(self, max_clients=16):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(max_clients, dtype=torch.float32))

    def forward(self, client_ids):
        return self.logits[client_ids]


class PairAttentionMLP(nn.Module):
    """
    Small attention network used by QAGAFed/FGL-AC style modes.
    It maps pairwise client features [x_i, x_j, |x_i-x_j|] to an edge logit.
    """
    def __init__(self, feat_dim=6, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xi, xj):
        z = torch.cat([xi, xj, torch.abs(xi - xj)], dim=-1)
        return self.net(z).squeeze(-1)


class SaveModelStrategy(FedAvg):
    def __init__(
        self,
        aggregation_mode: str = "fedavg",
        gamma: float = 1.0,
        epsilon: float = 1e-8,
        loss_metric_key: str = "val_loss",
        score_metric_key: str = "val_ssim",
        num_rounds: int = 400,
        server_lr: float = 0.1,
        fedadam_beta1: float = 0.9,
        fedadam_beta2: float = 0.99,
        fedadam_tau: float = 1e-9,
        nn_hidden_dim: int = 16,
        nn_lr: float = 1e-3,
        nn_warmup_rounds: int = 5,
        nn_temperature: float = 0.10,
        history_path: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latest_parameters = None
        self.current_ndarrays = None

        self.aggregation_mode = aggregation_mode
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_metric_key = loss_metric_key
        self.score_metric_key = score_metric_key
        self.num_rounds = max(num_rounds, 1)

        # FedAdam server optimizer states
        self.server_lr = server_lr
        self.fedadam_beta1 = fedadam_beta1
        self.fedadam_beta2 = fedadam_beta2
        self.fedadam_tau = fedadam_tau
        self.m_t = None
        self.v_t = None

        # NN aggregator
        self.nn_warmup_rounds = nn_warmup_rounds
        self.nn_temperature = nn_temperature
        self.weight_net = ClientWeightMLP(in_dim=6, hidden_dim=nn_hidden_dim)
        self.weight_optim = torch.optim.Adam(self.weight_net.parameters(), lr=nn_lr)

        # Additional server-side lightweight aggregation learners.
        # They do not touch client code or local pGAN training.
        self.fedlaw_net = FedLAWWeightModel(max_clients=32)
        self.fedlaw_optim = torch.optim.Adam(self.fedlaw_net.parameters(), lr=nn_lr)

        self.pair_attention = PairAttentionMLP(feat_dim=6, hidden_dim=nn_hidden_dim)
        self.pair_attention_optim = torch.optim.Adam(self.pair_attention.parameters(), lr=nn_lr)
        # ------------------------------------------------------------
        # RL / bandit-style aggregation state.
        # This does not need offline data. It starts from a safe policy
        # and updates weights using an online reward proxy.
        # ------------------------------------------------------------
        self.rl_policy_logits = None
        self.rl_baseline_reward = None
        self.rl_prev_weights = None
        self.rl_prev_reward = None

        self.rl_lr = float(os.environ.get("RL_AGG_LR", 0.10))
        self.rl_entropy = float(os.environ.get("RL_AGG_ENTROPY", 0.02))
        self.rl_exploration = float(os.environ.get("RL_AGG_EXPLORATION", 0.05))
        self.rl_warmup_rounds = int(
            os.environ.get("RL_AGG_WARMUP_ROUNDS", self.nn_warmup_rounds)
        )
        self.history_path = Path(history_path or OUTPUT_ROOT / "history" / f"{aggregation_mode}_history.jsonl")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize_parameters(self, client_manager):
        """
        Create deterministic initial global parameters on the server.
        This is important for FedAdam because it needs current global params
        to compute client deltas.
        """
        netG, netD = create_models(DEVICE, seed=int(os.environ.get("FL_SEED", 42)))
        init_ndarrays = get_combined_parameters(netG, netD)
        self.current_ndarrays = [arr.copy() for arr in init_ndarrays]
        self.latest_parameters = ndarrays_to_parameters(init_ndarrays)
        print("[SERVER] Initialized global G+D parameters on server.", flush=True)
        return self.latest_parameters

    def _weighted_average_params(self, client_ndarrays_list, weights):
        aggregated = None
        for client_ndarrays, alpha_k in zip(client_ndarrays_list, weights):
            if aggregated is None:
                aggregated = [alpha_k * arr for arr in client_ndarrays]
            else:
                for i in range(len(client_ndarrays)):
                    aggregated[i] += alpha_k * client_ndarrays[i]
        return aggregated

    def _extract_client_data(self, server_round, results):
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            raise ValueError("Total examples is zero.")

        rows = []
        client_ndarrays_list = []

        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            client_id = int(metrics.get("client_id", len(rows)))
            val_loss = float(metrics.get("val_loss", np.inf))
            val_ssim = float(metrics.get("val_ssim", 0.0))
            val_psnr = float(metrics.get("val_psnr", 0.0))
            n = int(fit_res.num_examples)
            local_steps = int(metrics.get("local_steps", 1))

            if not np.isfinite(val_loss):
                raise ValueError(f"Client {client_id} returned non-finite val_loss={val_loss}")

            rows.append(
                {
                    "client_id": client_id,
                    "num_examples": n,
                    "sample_share": n / total_examples,
                    "val_loss": val_loss,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr,
                    "round_progress": server_round / self.num_rounds,
                    "local_steps": max(local_steps, 1),
                }
            )
            client_ndarrays_list.append(parameters_to_ndarrays(fit_res.parameters))

        return rows, client_ndarrays_list

    def _weights_fedavg(self, rows):
        return np.array([r["sample_share"] for r in rows], dtype=np.float64)

    def _weights_inverse_loss(self, rows, gamma=None):
        gamma = self.gamma if gamma is None else gamma
        raw = []
        for r in rows:
            raw.append(r["sample_share"] * ((r["val_loss"] + self.epsilon) ** (-gamma)))
        raw = np.array(raw, dtype=np.float64)
        return raw / max(raw.sum(), self.epsilon)

    def _weights_softmax_score(self, rows):
        """
        Directly prioritizes higher client validation SSIM.
        Uses sample_share as a stabilizer so tiny sites do not dominate.
        """
        scores = np.array([r[self.score_metric_key] for r in rows], dtype=np.float64)
        shares = np.array([r["sample_share"] for r in rows], dtype=np.float64)
        scores = scores - np.max(scores)
        raw = np.exp(scores / max(self.nn_temperature, self.epsilon)) * shares
        return raw / max(raw.sum(), self.epsilon)

    def _features_tensor(self, rows):
        max_client_id = max(max(r["client_id"] for r in rows), 1)

        # Stable normalization inside a round
        losses = np.array([r["val_loss"] for r in rows], dtype=np.float64)
        psnrs = np.array([r["val_psnr"] for r in rows], dtype=np.float64)
        ssims = np.array([r["val_ssim"] for r in rows], dtype=np.float64)

        def z(v):
            return (v - v.mean()) / (v.std() + 1e-8)

        feats = []
        loss_z = z(losses)
        psnr_z = z(psnrs)
        ssim_z = z(ssims)

        for i, r in enumerate(rows):
            feats.append([
                r["sample_share"],
                loss_z[i],
                ssim_z[i],
                psnr_z[i],
                r["round_progress"],
                r["client_id"] / max_client_id,
            ])

        return torch.tensor(feats, dtype=torch.float32)

    def _weights_nn(self, server_round, rows):
        """
        NN-based node credit.
        Warmup: use SSIM-softmax weights.
        Training target: SSIM-softmax pseudo-labels.
        Reason: true final global SSIM is only known after aggregation, so online
        per-client labels are not directly available. This makes the NN learn a
        stable mapping from client stats to SSIM-oriented credit.
        """
        target_np = self._weights_softmax_score(rows)
        target = torch.tensor(target_np, dtype=torch.float32)

        x = self._features_tensor(rows)

        self.weight_net.train()
        logits = self.weight_net(x)
        pred = torch.softmax(logits, dim=0)
        loss = F.kl_div(torch.log(pred + 1e-8), target, reduction="batchmean")

        self.weight_optim.zero_grad()
        loss.backward()
        self.weight_optim.step()

        if server_round <= self.nn_warmup_rounds:
            weights = target_np
            print(
                f"[SERVER][NN] warmup round={server_round}; using SSIM-softmax target weights.",
                flush=True,
            )
        else:
            self.weight_net.eval()
            with torch.no_grad():
                weights = torch.softmax(self.weight_net(x), dim=0).cpu().numpy()

        print(f"[SERVER][NN] train_loss={float(loss.item()):.8f}", flush=True)
        return weights.astype(np.float64)

    def _safe_normalize(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.maximum(raw, 0.0)
        s = raw.sum()
        if s <= self.epsilon:
            return np.ones_like(raw, dtype=np.float64) / max(len(raw), 1)
        return raw / s

    def _quality_target_weights(self, rows):
        """
        Pseudo-label used to train server-side weighting networks online.
        It combines high SSIM, high PSNR, low loss, and sample share.
        This is safer than pure SSIM for pGAN because very small clients can
        occasionally show high SSIM by chance.
        """
        shares = np.array([r["sample_share"] for r in rows], dtype=np.float64)
        losses = np.array([r["val_loss"] for r in rows], dtype=np.float64)
        ssims = np.array([r["val_ssim"] for r in rows], dtype=np.float64)
        psnrs = np.array([r["val_psnr"] for r in rows], dtype=np.float64)

        def z(v):
            return (v - v.mean()) / (v.std() + 1e-8)

        # Larger score is better. Loss is subtracted.
        quality = 1.5 * z(ssims) + 0.5 * z(psnrs) - 1.0 * z(losses)
        quality = quality - np.max(quality)
        raw = np.exp(quality / max(self.nn_temperature, self.epsilon)) * np.sqrt(shares + self.epsilon)
        return self._safe_normalize(raw)

    def _client_deltas_flat(self, client_ndarrays_list, max_values_per_tensor=2048):
        """
        Compact flattened update vectors for graph similarity.
        We sample only the first values of each tensor to keep server memory safe.
        This is enough to estimate update direction similarity between 4 clients.
        """
        if self.current_ndarrays is None:
            return None

        flat = []
        for client_params in client_ndarrays_list:
            parts = []
            for p, g in zip(client_params, self.current_ndarrays):
                d = (p - g).reshape(-1)
                if d.size > max_values_per_tensor:
                    d = d[:max_values_per_tensor]
                parts.append(d.astype(np.float32, copy=False))
            flat.append(np.concatenate(parts))
        return np.stack(flat, axis=0)

    def _update_similarity_matrix(self, client_ndarrays_list):
        deltas = self._client_deltas_flat(client_ndarrays_list)
        k = len(client_ndarrays_list)
        if deltas is None or k == 0:
            return np.eye(k, dtype=np.float64)

        norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8
        normed = deltas / norms
        sim = normed @ normed.T
        sim = (sim + 1.0) / 2.0  # map cosine from [-1,1] to [0,1]
        np.fill_diagonal(sim, 1.0)
        return sim.astype(np.float64)

    def _weights_fedlaw(self, server_round, rows):
        """
        FedLAW-style learnable aggregation weights.
        The learnable client logits are trained online toward a quality-aware target.
        """
        client_ids = torch.tensor([int(r["client_id"]) for r in rows], dtype=torch.long)
        target_np = self._quality_target_weights(rows)
        target = torch.tensor(target_np, dtype=torch.float32)

        logits = self.fedlaw_net(client_ids)
        pred = torch.softmax(logits, dim=0)
        loss = F.kl_div(torch.log(pred + 1e-8), target, reduction="batchmean")

        self.fedlaw_optim.zero_grad()
        loss.backward()
        self.fedlaw_optim.step()

        if server_round <= self.nn_warmup_rounds:
            weights = target_np
            print(f"[SERVER][FedLAW] warmup round={server_round}; using quality target.", flush=True)
        else:
            with torch.no_grad():
                weights = torch.softmax(self.fedlaw_net(client_ids), dim=0).cpu().numpy()

        print(f"[SERVER][FedLAW] train_loss={float(loss.item()):.8f}", flush=True)
        return self._safe_normalize(weights)

    def _edge_attention_matrix(self, rows, client_ndarrays_list=None, use_update_similarity=True):
        """
        Builds a KxK graph attention matrix A where A_ij means how much client i
        attends to client j. It combines learned pairwise attention and optional
        model-update similarity.
        """
        x = self._features_tensor(rows)
        k = x.shape[0]
        logits = torch.zeros((k, k), dtype=torch.float32)

        for i in range(k):
            for j in range(k):
                logits[i, j] = self.pair_attention(x[i], x[j])

        if use_update_similarity and client_ndarrays_list is not None:
            sim_np = self._update_similarity_matrix(client_ndarrays_list)
            sim = torch.tensor(sim_np, dtype=torch.float32)
            logits = logits + torch.log(sim + 1e-6)

        A = torch.softmax(logits, dim=1)
        return A

    def _weights_qagafed(self, server_round, rows, client_ndarrays_list):
        """
        QAGAFed: quality-aware graph attention.
        Node features = sample share, loss, SSIM, PSNR, progress, client id.
        Edges = learned pair attention + update similarity.
        Client weight = quality target propagated through graph attention.
        """
        target_np = self._quality_target_weights(rows)
        target = torch.tensor(target_np, dtype=torch.float32)

        A = self._edge_attention_matrix(rows, client_ndarrays_list, use_update_similarity=True)
        propagated = torch.matmul(A.transpose(0, 1), target)
        pred = propagated / (propagated.sum() + 1e-8)

        # Train pair attention so propagated graph weights approximate quality target.
        loss = F.kl_div(torch.log(pred + 1e-8), target, reduction="batchmean")
        self.pair_attention_optim.zero_grad()
        loss.backward()
        self.pair_attention_optim.step()

        if server_round <= self.nn_warmup_rounds:
            weights = target_np
            print(f"[SERVER][QAGAFed] warmup round={server_round}; using quality target.", flush=True)
        else:
            with torch.no_grad():
                A = self._edge_attention_matrix(rows, client_ndarrays_list, use_update_similarity=True)
                propagated = torch.matmul(A.transpose(0, 1), target)
                weights = (propagated / (propagated.sum() + 1e-8)).cpu().numpy()

        print(f"[SERVER][QAGAFed] graph_loss={float(loss.item()):.8f}", flush=True)
        return self._safe_normalize(weights)

    def _weights_fgl_ac(self, server_round, rows):
        """
        FGL-AC-inspired attention aggregation.
        This implementation keeps only the useful attention component because
        with four hospitals, hard clustering is often unstable.
        """
        target_np = self._quality_target_weights(rows)
        x = self._features_tensor(rows)
        logits = self.weight_net(x)
        pred = torch.softmax(logits, dim=0)
        target = torch.tensor(target_np, dtype=torch.float32)

        loss = F.kl_div(torch.log(pred + 1e-8), target, reduction="batchmean")
        self.weight_optim.zero_grad()
        loss.backward()
        self.weight_optim.step()

        if server_round <= self.nn_warmup_rounds:
            weights = target_np
        else:
            with torch.no_grad():
                weights = torch.softmax(self.weight_net(x), dim=0).cpu().numpy()

        print(f"[SERVER][FGL-AC] attention_loss={float(loss.item()):.8f}", flush=True)
        return self._safe_normalize(weights)

    def _weights_fedga(self, rows, client_ndarrays_list):
        """
        FedGA-inspired graph aggregation.
        Builds graph by update similarity and computes graph centrality weighted
        by quality. More central high-quality updates get more credit.
        """
        quality = self._quality_target_weights(rows)
        sim = self._update_similarity_matrix(client_ndarrays_list)
        centrality = sim.mean(axis=1)
        raw = quality * (centrality + self.epsilon)
        return self._safe_normalize(raw)

    def _weights_gpfl(self, rows, client_ndarrays_list):
        """
        GPFL-inspired client-network learning.
        Since this experiment has one shared final pGAN, we convert the learned
        client graph into one global aggregation vector by averaging each client's
        personalized incoming graph weights.
        """
        quality = self._quality_target_weights(rows)
        sim = self._update_similarity_matrix(client_ndarrays_list)

        # Personalized graph: each row is a client-specific mixture over neighbors.
        G = sim * quality[None, :]
        G = G / (G.sum(axis=1, keepdims=True) + self.epsilon)

        # Convert personalized graph to one global aggregation vector.
        weights = G.mean(axis=0)
        return self._safe_normalize(weights)

    def _weights_hybrid_fedgraph(self, rows, client_ndarrays_list):
        """
        Hybrid FedGraph: convex combination of FedAvg and graph-quality weights.
        This is often safer with only 4 hospitals than a pure graph method.
        """
        fedavg = self._weights_fedavg(rows)
        graph = self._weights_fedga(rows, client_ndarrays_list)
        lam = 0.5
        return self._safe_normalize(lam * fedavg + (1.0 - lam) * graph)

    def _round_reward_proxy(self, rows):
        """
        Lightweight online reward proxy for RL aggregation.

        True final global SSIM is only available after expensive final evaluation.
        During training rounds, clients already return local validation metrics.
        This reward combines:
        + validation SSIM
        + validation PSNR
        - validation loss
        """
        ssims = np.array([r["val_ssim"] for r in rows], dtype=np.float64)
        psnrs = np.array([r["val_psnr"] for r in rows], dtype=np.float64)
        losses = np.array([r["val_loss"] for r in rows], dtype=np.float64)
        shares = np.array([r["sample_share"] for r in rows], dtype=np.float64)

        def z(v):
            return (v - v.mean()) / (v.std() + 1e-8)

        # Bigger is better.
        quality = 1.5 * z(ssims) + 0.5 * z(psnrs) - 1.0 * z(losses)

        # Server reward proxy for this round.
        reward = float(np.sum(shares * quality))
        return reward

    def _weights_rl_bandit(self, server_round, rows):
        """
        RL / bandit-style aggregation.

        It works without offline data:
        - early rounds use quality/SSIM target weights
        - later rounds update a small policy using reward improvement
        - exploration noise prevents collapse to one client
        """
        k = len(rows)

        # Use your existing quality target as a safe starting policy.
        target = self._quality_target_weights(rows)

        if self.rl_policy_logits is None or len(self.rl_policy_logits) != k:
            self.rl_policy_logits = np.log(target + self.epsilon).astype(np.float64)
            self.rl_prev_weights = target.copy()
            self.rl_prev_reward = None
            self.rl_baseline_reward = None

        reward = self._round_reward_proxy(rows)

        if self.rl_baseline_reward is None:
            self.rl_baseline_reward = reward
        else:
            self.rl_baseline_reward = 0.9 * self.rl_baseline_reward + 0.1 * reward

        # Warm-up phase: do not learn yet.
        # Use the quality target and collect reward history.
        if server_round <= self.rl_warmup_rounds or self.rl_prev_reward is None:
            weights = target
            self.rl_prev_weights = weights.copy()
            self.rl_prev_reward = reward

            print(
                f"[SERVER][RL] warmup round={server_round} | "
                f"reward_proxy={reward:.6f} | using quality target.",
                flush=True,
            )

            return self._safe_normalize(weights)

        # Advantage tells us whether the last policy was good or bad.
        advantage = reward - self.rl_baseline_reward

        current_policy = self._safe_normalize(
            np.exp(self.rl_policy_logits - np.max(self.rl_policy_logits))
        )

        # Policy-gradient-like update:
        # if advantage > 0, reinforce previous weights
        # if advantage < 0, move away from previous weights
        grad = self.rl_prev_weights - current_policy
        self.rl_policy_logits = self.rl_policy_logits + self.rl_lr * advantage * grad

        policy = self._safe_normalize(
            np.exp(self.rl_policy_logits - np.max(self.rl_policy_logits))
        )

        # Entropy smoothing prevents all weight going to one client too early.
        uniform = np.ones(k, dtype=np.float64) / k
        policy = (1.0 - self.rl_entropy) * policy + self.rl_entropy * uniform

        # Small exploration noise.
        noise = np.random.normal(loc=0.0, scale=self.rl_exploration, size=k)
        explored = self._safe_normalize(policy + noise)

        print(
            f"[SERVER][RL] round={server_round} | "
            f"reward_proxy={reward:.6f} | "
            f"baseline={self.rl_baseline_reward:.6f} | "
            f"advantage={advantage:.6f} | "
            f"weights={explored.tolist()}",
            flush=True,
        )

        self.rl_prev_weights = explored.copy()
        self.rl_prev_reward = reward

        return self._safe_normalize(explored)

    def _aggregate_fednova(self, client_ndarrays_list, weights, rows):
        """
        Practical FedNova-style normalized update aggregation.

        For each client:
          delta_k = client_params_k - current_global_params
          normalized_delta_k = delta_k / local_steps_k

        Server:
          delta = tau_eff * sum_k p_k * normalized_delta_k
          new_params = current_global_params + delta

        This matters most when clients perform different numbers of local steps.
        If local steps are nearly equal, FedNova will be very close to FedAvg.
        """
        if self.current_ndarrays is None:
            self.current_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        tau = np.array([max(float(r.get("local_steps", 1)), 1.0) for r in rows], dtype=np.float64)
        tau_eff = float(np.sum(weights * tau))

        new_params = []
        for p_idx in range(len(self.current_ndarrays)):
            normalized_delta = None
            for client_params, alpha, tau_k in zip(client_ndarrays_list, weights, tau):
                d = (client_params[p_idx] - self.current_ndarrays[p_idx]) / tau_k
                normalized_delta = alpha * d if normalized_delta is None else normalized_delta + alpha * d

            new_params.append(self.current_ndarrays[p_idx] + tau_eff * normalized_delta)

        self.current_ndarrays = [p.copy() for p in new_params]
        return new_params

    def _aggregate_fedadam(self, client_ndarrays_list, weights, server_round):
        if self.current_ndarrays is None:
            # Fallback: first round becomes weighted-average params.
            self.current_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        avg_delta = []
        for p_idx in range(len(self.current_ndarrays)):
            delta = None
            for client_params, alpha in zip(client_ndarrays_list, weights):
                d = client_params[p_idx] - self.current_ndarrays[p_idx]
                delta = alpha * d if delta is None else delta + alpha * d
            avg_delta.append(delta)

        if self.m_t is None:
            self.m_t = [np.zeros_like(d) for d in avg_delta]
            self.v_t = [np.zeros_like(d) for d in avg_delta]

        new_params = []
        for i, delta in enumerate(avg_delta):
            self.m_t[i] = self.fedadam_beta1 * self.m_t[i] + (1.0 - self.fedadam_beta1) * delta
            self.v_t[i] = self.fedadam_beta2 * self.v_t[i] + (1.0 - self.fedadam_beta2) * (delta * delta)

            m_hat = self.m_t[i] / (1.0 - self.fedadam_beta1 ** server_round)
            v_hat = self.v_t[i] / (1.0 - self.fedadam_beta2 ** server_round)

            new_p = self.current_ndarrays[i] + self.server_lr * m_hat / (np.sqrt(v_hat) + self.fedadam_tau)
            new_params.append(new_p)

        self.current_ndarrays = [p.copy() for p in new_params]
        return new_params

    def _save_round_history(self, server_round, rows, weights, mode):
        record = {
            "round": server_round,
            "aggregation_mode": mode,
            "weights": [float(w) for w in weights],
            "clients": rows,
        }
        with self.history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(
            f"[SERVER] Aggregating round {server_round} | success={len(results)} | failures={len(failures)}",
            flush=True,
        )

        if not results:
            print("[SERVER] No results received this round.", flush=True)
            return None, {}

        rows, client_ndarrays_list = self._extract_client_data(server_round, results)

        for r in rows:
            print(
                f"[SERVER] client={r['client_id']} | n={r['num_examples']} | "
                f"share={r['sample_share']:.4f} | loss={r['val_loss']:.6f} | "
                f"ssim={r['val_ssim']:.6f} | psnr={r['val_psnr']:.6f}",
                flush=True,
            )

        if self.aggregation_mode == "fedavg":
            weights = self._weights_fedavg(rows)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "loss_weighted":
            weights = self._weights_inverse_loss(rows, gamma=1.0)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "gamma_loss_weighted":
            weights = self._weights_inverse_loss(rows, gamma=self.gamma)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "ssim_weighted":
            weights = self._weights_softmax_score(rows)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "nn_weighted":
            weights = self._weights_nn(server_round, rows)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "fedlaw":
            weights = self._weights_fedlaw(server_round, rows)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "qagafed":
            weights = self._weights_qagafed(server_round, rows, client_ndarrays_list)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "fgl_ac":
            weights = self._weights_fgl_ac(server_round, rows)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "fedga":
            weights = self._weights_fedga(rows, client_ndarrays_list)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "gpfl":
            weights = self._weights_gpfl(rows, client_ndarrays_list)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "hybrid_fedgraph":
            weights = self._weights_hybrid_fedgraph(rows, client_ndarrays_list)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "rl_bandit":
            weights = self._weights_rl_bandit(server_round, rows)
            aggregated_ndarrays = self._weighted_average_params(client_ndarrays_list, weights)

        elif self.aggregation_mode == "fednova":
            weights = self._weights_fedavg(rows)
            aggregated_ndarrays = self._aggregate_fednova(client_ndarrays_list, weights, rows)

        elif self.aggregation_mode == "fedadam":
            # FedAdam uses FedAvg-style client weighting for the update direction.
            weights = self._weights_fedavg(rows)
            aggregated_ndarrays = self._aggregate_fedadam(client_ndarrays_list, weights, server_round)

        elif self.aggregation_mode == "fedadam_ssim":
            # FedAdam, but client update directions are weighted by client validation SSIM.
            weights = self._weights_softmax_score(rows)
            aggregated_ndarrays = self._aggregate_fedadam(client_ndarrays_list, weights, server_round)

        else:
            raise ValueError(
                f"Unknown aggregation_mode='{self.aggregation_mode}'. "
                "Use one of: fedavg, loss_weighted, gamma_loss_weighted, "
                "ssim_weighted, nn_weighted, fedlaw, qagafed, fgl_ac, "
                "fedga, gpfl, hybrid_fedgraph, fednova, fedadam, fedadam_ssim"
            )

        weights = np.array(weights, dtype=np.float64)
        weights = weights / max(weights.sum(), self.epsilon)

        print(f"[SERVER] Normalized aggregation weights: {weights.tolist()}", flush=True)

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        self.latest_parameters = aggregated_parameters

        if self.aggregation_mode not in ["fednova", "fedadam", "fedadam_ssim"]:
            self.current_ndarrays = [arr.copy() for arr in aggregated_ndarrays]

        self._save_round_history(server_round, rows, weights, self.aggregation_mode)

        aggregated_metrics = {
            "aggregation_mode": self.aggregation_mode,
            "gamma": float(self.gamma),
            "loss_metric_key": self.loss_metric_key,
            "score_metric_key": self.score_metric_key,
        }

        print(
            f"[SERVER] Finished aggregation for round {server_round} | mode={self.aggregation_mode}",
            flush=True,
        )
        return aggregated_parameters, aggregated_metrics

def run_centralized_training(args):
    """
    Centralized baseline using selected FL client/site IDs.

    This does not start Flower.
    It trains one shared model sequentially over the selected client datasets.
    In practice, this simulates training on pooled sites while using the existing
    site-specific dataloading/training logic.
    """

    print("[CENTRAL] Starting centralized training baseline.", flush=True)
    print(f"[CENTRAL] Task name: {args.task_name}", flush=True)
    print(f"[CENTRAL] Num selected sites: {args.num_clients}", flush=True)
    print(f"[CENTRAL] Epochs: {args.num_rounds}", flush=True)
    print(f"[CENTRAL] Seed: {args.seed}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    netG, netD = create_models(DEVICE, seed=args.seed)
    global_params = get_combined_parameters(netG, netD)

    history_path = Path(
        args.history_path
        or OUTPUT_ROOT / "history" / f"centralized_{args.task_name}_history.jsonl"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.num_rounds + 1):
        print(f"[CENTRAL] Epoch {epoch}/{args.num_rounds}", flush=True)

        epoch_records = []

        for client_id in range(args.num_clients):
            print(
                f"[CENTRAL] Training on selected site/client {client_id}",
                flush=True,
            )

            # Reuse the same model parameters and continue training site by site.
            # This is centralized/sequential training, not federated aggregation.
            updated_parameters, num_examples, metrics = train_local_client(
                args.task_name,
                client_id,
                global_params,
                args.local_epochs,
            )

            global_params = updated_parameters

            if not isinstance(metrics, dict):
                metrics = {"val_loss": float(metrics)}

            record = {
                "epoch": epoch,
                "client_id": client_id,
                "num_examples": int(num_examples),
                "task_name": args.task_name,
                "run_mode": "centralized",
                "metrics": metrics,
            }

            epoch_records.append(record)

            with history_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

            print(
                f"[CENTRAL] site={client_id} | "
                f"loss={metrics.get('val_loss', None)} | "
                f"ssim={metrics.get('val_ssim', None)} | "
                f"psnr={metrics.get('val_psnr', None)}",
                flush=True,
            )

        if epoch % 10 == 0 or epoch == args.num_rounds:
            ckpt_dir = OUTPUT_ROOT / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            ckpt_path = ckpt_dir / f"centralized_{args.task_name}_epoch_{epoch}.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "task_name": args.task_name,
                    "seed": args.seed,
                    "global_parameters": global_params,
                },
                ckpt_path,
            )

            print(f"[CENTRAL] Saved checkpoint: {ckpt_path}", flush=True)

    print("[CENTRAL] Running final centralized evaluation...", flush=True)

    final_results = evaluate_global(args.task_name, global_params)

    print(f"[CENTRAL] FINAL EVAL RESULTS: {final_results}", flush=True)

    final_path = OUTPUT_ROOT / "history" / f"centralized_{args.task_name}_final.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    with final_path.open("w") as f:
        json.dump(final_results, f, indent=2)

    print(f"[CENTRAL] Saved final results to: {final_path}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_mode",
        type=str,
        default="fl",
        choices=["fl", "centralized"],
        help="Use 'fl' for federated learning or 'centralized' for centralized training baseline.",
    )
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=4)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--grpc_max_message_length", type=int, default=2147483647)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="fedavg",
        choices=[
            "fedavg",
            "loss_weighted",
            "gamma_loss_weighted",
            "ssim_weighted",
            "nn_weighted",
            "fedlaw",
            "qagafed",
            "fgl_ac",
            "fedga",
            "gpfl",
            "hybrid_fedgraph",
            "rl_bandit",
            "fednova",
            "fedadam",
            "fedadam_ssim",
        ],
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--loss_metric_key", type=str, default="val_loss")
    parser.add_argument("--score_metric_key", type=str, default="val_ssim")

    # FedAdam
    parser.add_argument("--server_lr", type=float, default=0.1)
    parser.add_argument("--fedadam_beta1", type=float, default=0.9)
    parser.add_argument("--fedadam_beta2", type=float, default=0.99)
    parser.add_argument("--fedadam_tau", type=float, default=1e-9)

    # NN aggregator
    parser.add_argument("--nn_hidden_dim", type=int, default=16)
    parser.add_argument("--nn_lr", type=float, default=1e-3)
    parser.add_argument("--nn_warmup_rounds", type=int, default=5)
    parser.add_argument("--nn_temperature", type=float, default=0.10)
    # RL/bandit aggregator hyperparameters
    parser.add_argument("--rl_lr", type=float, default=0.10)
    parser.add_argument("--rl_entropy", type=float, default=0.02)
    parser.add_argument("--rl_exploration", type=float, default=0.05)
    parser.add_argument("--rl_warmup_rounds", type=int, default=None)

    parser.add_argument("--history_path", type=str, default=None)

    args = parser.parse_args()

    os.environ["FL_SEED"] = str(args.seed)

    os.environ["RL_AGG_LR"] = str(args.rl_lr)
    os.environ["RL_AGG_ENTROPY"] = str(args.rl_entropy)
    os.environ["RL_AGG_EXPLORATION"] = str(args.rl_exploration)

    if args.rl_warmup_rounds is not None:
        os.environ["RL_AGG_WARMUP_ROUNDS"] = str(args.rl_warmup_rounds)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[SERVER] Seed: {args.seed}", flush=True)

    if args.num_clients is None:
        args.num_clients = get_num_site_clients(args.task_name)

    print(f"[SERVER] Number of clients/sites: {args.num_clients}", flush=True)
    print(f"[SERVER] Task name: {args.task_name}", flush=True)
    print(f"[SERVER] Aggregation mode: {args.aggregation_mode}", flush=True)


    if args.run_mode == "centralized":
        run_centralized_training(args)
        return

    def fit_config(server_round: int):
        return {
            "local_epochs": args.local_epochs,
            "server_round": server_round,
            "task_name": args.task_name,
        }

    strategy = SaveModelStrategy(
        aggregation_mode=args.aggregation_mode,
        gamma=args.gamma,
        epsilon=args.epsilon,
        loss_metric_key=args.loss_metric_key,
        score_metric_key=args.score_metric_key,
        num_rounds=args.num_rounds,
        server_lr=args.server_lr,
        fedadam_beta1=args.fedadam_beta1,
        fedadam_beta2=args.fedadam_beta2,
        fedadam_tau=args.fedadam_tau,
        nn_hidden_dim=args.nn_hidden_dim,
        nn_lr=args.nn_lr,
        nn_warmup_rounds=args.nn_warmup_rounds,
        nn_temperature=args.nn_temperature,
        history_path=args.history_path,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        on_fit_config_fn=fit_config,
    )

    print(
        f"[SERVER] Starting Flower server at {args.server_address} | rounds={args.num_rounds}",
        flush=True,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        grpc_max_message_length=args.grpc_max_message_length,
    )

    if strategy.latest_parameters is not None:
        print("[SERVER] Running final global evaluation...", flush=True)
        final_ndarrays = parameters_to_ndarrays(strategy.latest_parameters)
        final_results = evaluate_global(args.task_name, final_ndarrays)
        print(f"[SERVER] FINAL EVAL RESULTS: {final_results}", flush=True)
    else:
        print("[SERVER] No aggregated parameters found for final evaluation", flush=True)


if __name__ == "__main__":
    main()
