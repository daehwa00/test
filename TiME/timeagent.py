import torch
import torch.nn as nn
import torch.nn.functional as F
from TiME.traj_encoder import MambaTrajEncoder
import math
from typing import Dict


class TiMEAgent(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
        lambda_: float,
        epsilon: float,
        device: str,
        ef_enabled: bool = True,
        mr_enabled: bool = True,
        reset_dt: float = 5.0,
    ):
        super(TiMEAgent, self).__init__()

        self.policy_encoder = MambaTrajEncoder(
            tstep_dim=obs_size,
            max_seq_len=1000,
            d_model=256,
            d_state=64,
            d_conv=4,
            expand=2,
            n_layers=2,
            norm="layer",
            ef_enabled=ef_enabled,
            mr_enabled=mr_enabled,
            reset_dt=reset_dt,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_size * 2),
        )

        self.policy_hidden_state = None

        self.value_encoder = MambaTrajEncoder(
            tstep_dim=obs_size,
            max_seq_len=1000,
            d_model=256,
            d_state=64,
            d_conv=4,
            expand=2,
            n_layers=2,
            norm="layer",
            ef_enabled=ef_enabled,
            mr_enabled=mr_enabled,
            reset_dt=reset_dt,
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(obs_size, device=device)
        self.running_variance = torch.zeros(obs_size, device=device)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.device = device

    @torch.jit.export
    def policy(self, observation, resets=None, rollout=True):
        if rollout:
            with torch.no_grad():
                encoded, self.policy_hidden_state = self.policy_encoder(
                    observation, self.policy_hidden_state
                )
                return self.policy_head(encoded)
        else:
            encoded, _ = self.policy_encoder(observation, resets=resets)
            return self.policy_head(encoded)

    @torch.jit.export
    def value(self, observation, resets=None):
        encoded, _ = self.value_encoder(observation, resets=resets)
        return self.value_head(encoded)

    @torch.jit.export
    def dist_create(self, logits):
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + 0.001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(self, loc, scale):
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x):
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(self, observation, rollout=True):
        observation = self.normalize(observation)
        logits = self.policy(observation.unsqueeze(1), rollout=rollout)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
        truncation_mask = (~truncation).to(torch.float32)
        termination_mask = (~termination).to(torch.float32)
        values_t_plus_1 = torch.cat(
            [values[:, 1:], bootstrap_value.unsqueeze(1)], dim=1
        )
        deltas = reward + self.discounting * termination_mask * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)
        for ti in range(truncation_mask.shape[1] - 1, -1, -1):
            acc = (
                deltas[:, ti]
                + self.discounting
                * termination_mask[:, ti]
                * truncation_mask[:, ti]
                * self.lambda_
                * acc
            )
            vs_minus_v_xs[:, ti] = acc
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[:, 1:], bootstrap_value.unsqueeze(1)], dim=1)
        advantages = (
            reward + self.discounting * termination_mask * vs_t_plus_1 - values
        ) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor]):
        self.train()
        observation = self.normalize(td["observation"])
        if observation.ndim != 3 or td["done"].ndim != 2:
            raise ValueError("loss expects batch-major observation [M, T+1, O] and done [M, T]")
        if observation.shape[:2] != (td["done"].shape[0], td["done"].shape[1] + 1):
            raise ValueError("observation and done sequence dimensions are misaligned")

        policy_resets = torch.cat(
            (td["done"].new_zeros((td["done"].shape[0], 1)), td["done"][:, :-1]),
            dim=1,
        )
        value_resets = torch.cat(
            (td["done"].new_zeros((td["done"].shape[0], 1)), td["done"]), dim=1
        )
        policy_logits = self.policy(
            observation[:, :-1], rollout=False, resets=policy_resets
        )
        baseline = self.value(observation, resets=value_resets).squeeze(dim=-1)

        bootstrap_value = baseline[:, -1]
        baseline = baseline[:, :-1]

        reward = td["reward"] * self.reward_scaling
        termination = td["done"] & ~td["truncation"]

        loc, scale = self.dist_create(td["logits"])
        behaviour_action_log_probs = self.dist_log_prob(loc, scale, td["action"])
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, td["action"])
        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td["truncation"],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        not_done_mask = (~td["done"].bool()).float()

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages * not_done_mask
        surrogate_loss2 = (
            rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages * not_done_mask
        )
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        v_error = (vs - baseline) * not_done_mask
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        entropy = torch.mean(self.dist_entropy(loc, scale) * not_done_mask)
        entropy_loss = self.entropy_cost * -entropy

        total_loss = policy_loss + v_loss + entropy_loss

        return (
            total_loss,
            policy_loss.detach(),
            v_loss.detach(),
            entropy_loss.detach(),
        )
