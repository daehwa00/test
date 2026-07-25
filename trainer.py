from collections import namedtuple
from typing import Callable, Mapping

import torch

from env_utils import create_env
from TiME import TiMEAgent
from TiME.delta_telemetry import collect_policy_delta_telemetry
from TiME.provenance import (
    assert_shared_tensors_equal,
    backbone_provenance,
    copy_shared_tensors,
    shared_initialization_digest,
    refresh_memory_rebalancer_buffers,
)
from utils import derive_seed, make_generator, set_seed


StepData = namedtuple(
    "StepData", ("observation", "logits", "action", "reward", "done", "truncation")
)
EPISODE_LENGTH = 1000


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    return StepData(**{
        key: f(*[sd._asdict()[key] for sd in sds])
        for key in sds[0]._asdict()
    })


def _binary_mask(value: torch.Tensor, name: str) -> torch.Tensor:
    """Returns a Boolean reset mask, rejecting ambiguous wrapper output."""
    if value.dtype == torch.bool:
        return value
    if value.is_floating_point() or value.dtype in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }:
        if not torch.all((value == 0) | (value == 1)):
            raise ValueError(f"{name} must contain only 0/1 values")
        return value.to(dtype=torch.bool)
    raise TypeError(f"{name} must be Boolean or numeric 0/1, got {value.dtype}")


def _named_squeeze(value: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if value.shape[dim] != 1:
        raise ValueError(f"{name} expected singleton dimension {dim}, got {value.shape}")
    return value.squeeze(dim=dim)


def _record(config: Mapping, event: dict) -> None:
    recorder = config.get("history_recorder")
    if recorder is not None:
        recorder(event)


def _validate_rollout(td: StepData) -> None:
    if td.observation.dtype != torch.float32:
        raise TypeError(f"observation must be float32, got {td.observation.dtype}")
    for name in ("logits", "action", "reward"):
        value = getattr(td, name)
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
    for name in ("done", "truncation"):
        if getattr(td, name).dtype != torch.bool:
            raise TypeError(f"{name} must be bool")
    if td.observation.shape[0] != td.done.shape[0] + 1:
        raise ValueError("observation must contain exactly one bootstrap timestep")


def _evaluation_schedule(num_updates: int, evaluations: int) -> set[int]:
    if evaluations < 2:
        raise ValueError("evaluation_points must be at least 2")
    return {(index * num_updates) // (evaluations - 1) for index in range(evaluations)}


def evaluate_deterministic(
    agent, env, episode_count: int, episode_length: int = EPISODE_LENGTH
) -> list[float]:
    """Collect exactly one first episode per vector lane using tanh(loc)."""
    if episode_count != env.num_envs:
        raise ValueError("evaluation episode count must equal the vector batch size")
    was_training = agent.training
    previous_hidden_state = agent.policy_hidden_state
    returns = torch.zeros(env.num_envs, device=agent.device, dtype=torch.float32)
    active = torch.ones(env.num_envs, device=agent.device, dtype=torch.bool)
    agent.eval()
    agent.policy_hidden_state = agent.policy_encoder.init_hidden_state(env.num_envs, agent.device)
    observation = env.reset()
    try:
        with torch.no_grad():
            for _ in range(episode_length):
                logits = agent.policy(
                    agent.normalize(observation).unsqueeze(1), rollout=True
                )
                loc, _ = agent.dist_create(logits)
                action = _named_squeeze(torch.tanh(loc), 1, "evaluation action")
                observation, reward, done, _ = env.step(action)
                done = _binary_mask(done, "evaluation done")
                returns += reward.to(dtype=torch.float32) * active.to(dtype=torch.float32)
                active &= ~done
                if not active.any():
                    break
    finally:
        agent.policy_hidden_state = previous_hidden_state
        agent.train(was_training)
    if active.any():
        raise RuntimeError("evaluation did not complete one episode per vector lane")
    return [float(value) for value in returns.detach().cpu().tolist()]


def _record_evaluation(
    config: Mapping,
    agent,
    env,
    total_steps: int,
    evaluation_kind: str,
) -> None:
    episode_returns = evaluate_deterministic(
        agent,
        env,
        env.num_envs,
        episode_length=int(config.get("episode_length", EPISODE_LENGTH)),
    )
    _record(config, {
        "event": "evaluation",
        "step": int(total_steps),
        "total_steps": int(total_steps),
        "evaluation_kind": evaluation_kind,
        "evaluation_episodes": int(env.num_envs),
        "episode_return": episode_returns,
        "evaluation_mean_return": sum(episode_returns) / len(episode_returns),
    })


def _build_matched_agent(agent_kwargs: dict, config: Mapping, env_name: str, seed: int):
    """Construct a treatment and prove its common tensors match vanilla Mamba-2."""
    init_seed = derive_seed(env_name, seed, "shared-model-initialization")
    selected = {
        "ef_enabled": bool(config.get("ef_enabled", True)),
        "mr_enabled": bool(config.get("mr_enabled", True)),
        "reset_dt": float(config.get("reset_dt", 5.0)),
    }
    device = torch.device(agent_kwargs["device"])
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    def construct(**controls):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(init_seed)
            return TiMEAgent(**agent_kwargs, **controls)

    agent = construct(**selected)
    vanilla = construct(ef_enabled=False, mr_enabled=False, reset_dt=5.0)
    shared_names = copy_shared_tensors(vanilla, agent)
    rebalanced_buffer_count = refresh_memory_rebalancer_buffers(agent)
    assert_shared_tensors_equal(vanilla, agent)
    digest = shared_initialization_digest((vanilla, agent), shared_names)
    provenance = backbone_provenance()
    del vanilla
    return agent.to(device), {
        "step": 0,
        "event": "shared_initialization_verified",
        "shared_tensor_count": len(shared_names),
        "shared_initialization_digest": digest,
        "initialization_seed": init_seed,
        "rebalanced_buffer_count": rebalanced_buffer_count,
        "backbone_provenance": provenance,
    }

def train(config, env_name):
    protocol = config.get("protocol", "time-brax-corrected-v2")
    seed = int(config["seed"])
    set_seed(seed)
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="TiME", config=config)

    num_envs = config["num_envs"]
    device = config["device"]
    num_timesteps = config["num_timesteps"]
    unroll_length = config["unroll_length"]
    num_minibatches = config["num_minibatches"]
    num_update_epochs = config["num_update_epochs"]
    reward_scaling = config["reward_scaling"]
    entropy_cost = config["entropy_cost"]
    discounting = config["discounting"]
    learning_rate = config["learning_rate"]
    lambda_ = config["lambda_"]
    epsilon = config["epsilon"]
    episode_length = config.get("episode_length", EPISODE_LENGTH)
    if isinstance(episode_length, bool) or not isinstance(episode_length, int):
        raise TypeError("episode_length must be a positive integer")
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    keep_dims_override = config.get("keep_dims_override")
    if num_envs % num_minibatches:
        raise ValueError("num_envs must be divisible by num_minibatches")

    env_seed = derive_seed(env_name, seed, "training-brax")
    env = create_env(
        env_name,
        device,
        seed=env_seed,
        num_envs=num_envs,
        episode_length=episode_length,
        keep_dims_override=keep_dims_override,
    )

    env.reset()
    action = torch.zeros(env.action_space.shape, device=device)
    env.step(action)

    agent_kwargs = dict(
        obs_size=env.observation_space.shape[-1],
        action_size=env.action_space.shape[-1],
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        lambda_=lambda_,
        epsilon=epsilon,
        device=device,
    )
    # A disposable vanilla counterpart proves treatment-common initialization
    # before optimizer creation or any scientific forward.
    agent, initialization_event = _build_matched_agent(
        agent_kwargs, config, env_name, seed
    )
    _record(config, initialization_event)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    evaluation_points = int(config.get("evaluation_points", 6))
    intermediate_evaluation_episodes = int(
        config.get("evaluation_intermediate_episodes", 16)
    )
    final_evaluation_episodes = int(config.get("evaluation_final_episodes", 64))
    delta_telemetry_enabled = bool(config.get("delta_telemetry_enabled", False))
    delta_telemetry_points = int(config.get("delta_telemetry_points", 6))
    delta_telemetry_probe_sequences = int(
        config.get("delta_telemetry_probe_sequences", 8)
    )
    if delta_telemetry_enabled:
        if delta_telemetry_points < 2:
            raise ValueError("delta_telemetry_points must be at least 2")
        if not 1 <= delta_telemetry_probe_sequences <= num_envs:
            raise ValueError(
                "delta_telemetry_probe_sequences must be between 1 and num_envs"
            )
    if intermediate_evaluation_episodes <= 0 or final_evaluation_episodes <= 0:
        raise ValueError("evaluation episode counts must be positive")
    intermediate_evaluation_env = create_env(
        env_name,
        device,
        seed=derive_seed(env_name, seed, "evaluation-brax-intermediate"),
        num_envs=intermediate_evaluation_episodes,
        episode_length=episode_length,
        keep_dims_override=keep_dims_override,
    )
    final_evaluation_env = create_env(
        env_name,
        device,
        seed=derive_seed(env_name, seed, "evaluation-brax-final"),
        num_envs=final_evaluation_episodes,
        episode_length=episode_length,
        keep_dims_override=keep_dims_override,
    )

    num_steps = num_envs * unroll_length
    num_epochs = num_timesteps // num_steps
    minibatch_generator = make_generator(
        derive_seed(env_name, seed, "minibatch-order"), device
    )
    total_steps = 0
    evaluation_schedule = _evaluation_schedule(num_epochs, evaluation_points)
    delta_telemetry_schedule = (
        _evaluation_schedule(num_epochs, delta_telemetry_points)
        if delta_telemetry_enabled
        else set()
    )
    delta_telemetry_probe = None
    if 0 in evaluation_schedule:
        _record_evaluation(
            config,
            agent,
            intermediate_evaluation_env,
            total_steps,
            "intermediate",
        )
    for epoch in range(num_epochs):
        loss_totals = torch.zeros(4, device=device, dtype=torch.float32)
        _, td, episode_returns = train_unroll(agent, env, unroll_length, total_steps, config)
        agent.update_normalization(td.observation)
        if delta_telemetry_enabled and delta_telemetry_probe is None:
            delta_telemetry_probe = (
                agent.normalize(
                    td.observation[:-1, :delta_telemetry_probe_sequences]
                )
                .swapaxes(0, 1)
                .contiguous()
                .detach()
                .clone()
            )
            if 0 in delta_telemetry_schedule:
                _record(
                    config,
                    collect_policy_delta_telemetry(
                        agent, delta_telemetry_probe, total_steps
                    ),
                )

        for _ in range(num_update_epochs):
            with torch.no_grad():
                permutation = torch.randperm(
                    td.observation.shape[1], device=device, generator=minibatch_generator
                )

                def shuffle_batch(data):
                    data = data[:, permutation]
                    data = data.reshape(
                        [data.shape[0], num_minibatches, -1] + list(data.shape[2:])
                    )
                    return data.swapaxes(0, 1).contiguous()

                epoch_td = sd_map(shuffle_batch, td)
            for minibatch_i in range(num_minibatches):
                td_minibatch = sd_map(lambda data: data[minibatch_i], epoch_td)
                batch_major_td = sd_map(
                    lambda data: data.swapaxes(0, 1).contiguous(), td_minibatch
                )
                loss, policy_loss, value_loss, entropy_loss = agent.loss(
                    batch_major_td._asdict()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_totals += torch.stack((
                    loss.detach(),
                    policy_loss,
                    value_loss,
                    entropy_loss,
                ))

        total_steps += num_steps
        if (epoch + 1) in delta_telemetry_schedule:
            _record(
                config,
                collect_policy_delta_telemetry(
                    agent, delta_telemetry_probe, total_steps
                ),
            )
        if (epoch + 1) in evaluation_schedule:
            is_final_evaluation = epoch + 1 == num_epochs
            _record_evaluation(
                config,
                agent,
                final_evaluation_env
                if is_final_evaluation
                else intermediate_evaluation_env,
                total_steps,
                "final" if is_final_evaluation else "intermediate",
            )
        epoch_metrics = torch.cat((
            loss_totals,
            (td.reward.float().mean() * episode_length).reshape(1),
        )).detach().cpu().tolist()
        diagnostic = {
            "step": int(total_steps),
            "train/epoch_loss": epoch_metrics[0],
            "train/epoch_policy_loss": epoch_metrics[1],
            "train/epoch_value_loss": epoch_metrics[2],
            "train/epoch_entropy_loss": epoch_metrics[3],
            "training_unroll_reward": epoch_metrics[4],
        }
        if episode_returns:
            diagnostic["episode_return"] = episode_returns
        _record(config, diagnostic)
        if use_wandb:
            wandb.log(diagnostic, step=total_steps)
        else:
            print(f"[Train] Epoch {epoch}, Loss: {epoch_metrics[0]:.3f}")

    return {"protocol": protocol, "total_steps": total_steps}


def train_unroll(agent, env, unroll_length, step, config):
    agent.eval()
    observation = env.reset()
    agent.policy_hidden_state = agent.policy_encoder.init_hidden_state(env.num_envs, agent.device)
    rollout = StepData([observation], [], [], [], [], [])
    episode_running = torch.zeros(env.num_envs, device=agent.device, dtype=torch.float32)
    episode_returns = []
    completed_returns = []

    for _ in range(unroll_length):
        with torch.no_grad():
            logits, action = agent.get_logits_action(observation)
        environment_action = _named_squeeze(
            TiMEAgent.dist_postprocess(action), 1, "sampled action"
        )
        observation, reward, done, info = env.step(environment_action)
        done = _binary_mask(done, "done")
        truncation = _binary_mask(info["truncation"], "truncation")
        agent.policy_hidden_state.reset_state(done)
        episode_running += reward.to(dtype=torch.float32)
        completed_returns.append(
            torch.where(done, episode_running, torch.full_like(episode_running, torch.nan))
        )
        episode_running = torch.where(done, torch.zeros_like(episode_running), episode_running)
        rollout.observation.append(observation)
        rollout.logits.append(_named_squeeze(logits, 1, "policy logits"))
        rollout.action.append(_named_squeeze(action, 1, "sampled action"))
        rollout.reward.append(reward.to(dtype=torch.float32))
        rollout.done.append(done)
        rollout.truncation.append(truncation)

    # Storage remains time-major; the trainer performs the sole explicit
    # batch-major conversion immediately before TiMEAgent.loss.
    td = sd_map(lambda values: torch.stack(values, dim=0).contiguous(), rollout)
    _validate_rollout(td)
    completed = torch.stack(completed_returns)
    episode_returns = [
        float(value)
        for value in completed[torch.isfinite(completed)].detach().cpu().tolist()
    ]
    return observation, td, episode_returns
