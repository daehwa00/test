import gym
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax import envs

ENVS_DIM = {
    "halfcheetah": {
        "keep_dims": list(range(0, 9)),
    },
    "hopper": {
        "keep_dims": list(range(0, 5)),
    },
    "swimmer": {
        "keep_dims": list(range(0, 4)),
    },
    "reacher": {
        "keep_dims": [0, 1, 2, 3, 8, 9, 10],
    },
    "ant": {
        "keep_dims": list(range(0, 13)),
    },
    "humanoidstandup": {
        "keep_dims": list(range(0, 16)),
    },
    "pusher": {
        "keep_dims": list(range(0, 7)) + [14, 15, 16, 17, 18, 19],
    },
    "walker2d": {
        "keep_dims": [0, 2, 3, 4, 5, 6, 7],
    },
}


ENV_HYPERPARAMS = {
    "halfcheetah": {
        "discounting": 0.98,
        "lambda_": 0.92,
        "num_envs": 64,
        "unroll_length": 256,
        "entropy_cost": 0.0005558684366667077,
        "learning_rate": 0.00136509431273925,
        "reward_scaling": 0.32044334240011263,
        "num_timesteps": 3_000_000,
    },
    "hopper": {
        "discounting": 0.99,
        "lambda_": 0.95,
        "num_envs": 64,
        "unroll_length": 128,
        "entropy_cost": 0.00001430586420329152,
        "learning_rate": 0.00010018555363337202,
        "reward_scaling": 0.13706167636580946,
        "num_timesteps": 1_000_000,
    },
    "ant": {
        "discounting": 0.98,
        "lambda_": 0.93,
        "num_envs": 64,
        "unroll_length": 128,
        "entropy_cost": 0.00003698473904078169,
        "learning_rate": 0.0004037540234187324,
        "reward_scaling": 0.38578535484188625,
        "num_timesteps": 3_000_000,
    },
    "walker2d": {
        "discounting": 0.995,
        "lambda_": 0.98,
        "num_envs": 32,
        "unroll_length": 128,
        "entropy_cost": 0.00003066781225197179,
        "learning_rate": 0.00021918420087285032,
        "reward_scaling": 0.2281923341486558,
        "num_timesteps": 1_000_000,
    },
    "swimmer": {
        "discounting": 0.98,
        "lambda_": 0.99,
        "num_envs": 32,
        "unroll_length": 64,
        "entropy_cost": 0.00003430319237955782,
        "learning_rate": 0.0012626482323111997,
        "reward_scaling": 1,
        "num_timesteps": 1_000_000,
    },
    "reacher": {
        "discounting": 0.995,
        "lambda_": 0.98,
        "num_envs": 32,
        "unroll_length": 64,
        "entropy_cost": 0.00023869519228979816,
        "learning_rate": 0.0005078717564188115,
        "reward_scaling": 0.12877286712431638,
        "num_timesteps": 1_000_000,
    },
    "pusher": {
        "discounting": 0.99,
        "lambda_": 0.95,
        "num_envs": 32,
        "unroll_length": 128,
        "entropy_cost": 0.0009387007322881968,
        "learning_rate": 0.00037492908011401826,
        "reward_scaling": 0.26359850133423257,
        "num_timesteps": 1_000_000,
    },
    "humanoidstandup": {
        "discounting": 0.99,
        "lambda_": 0.8,
        "num_envs": 32,
        "unroll_length": 256,
        "entropy_cost": 0.00001250166915452103,
        "learning_rate": 0.0006849469830690655,
        "reward_scaling": 0.17345286822592065,
        "num_timesteps": 1_000_000,
    },
}


class POMDPWrapper(gym_wrapper.VectorGymWrapper):
    def __init__(self, env, keep_dims, seed=None):
        super().__init__(env, seed=seed)
        self.keep_dims = keep_dims
        low = self.observation_space.low[..., self.keep_dims]
        high = self.observation_space.high[..., self.keep_dims]
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def _exclude_dim(self, obs):
        return obs[..., self.keep_dims]

    def reset(self):
        obs = super().reset()
        return self._exclude_dim(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self._exclude_dim(obs), reward, done, info


def _validate_episode_length(episode_length):
    if isinstance(episode_length, bool) or not isinstance(episode_length, int):
        raise TypeError("episode_length must be a positive integer")
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    return episode_length


def _validate_keep_dims(keep_dims):
    if not isinstance(keep_dims, (list, tuple)):
        raise TypeError("keep_dims_override must be a sequence of integers")
    if not keep_dims:
        raise ValueError("keep_dims_override must not be empty")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in keep_dims):
        raise ValueError(
            "keep_dims_override must contain unique nonnegative integers"
        )
    if len(set(keep_dims)) != len(keep_dims):
        raise ValueError("keep_dims_override must contain unique nonnegative integers")
    return list(keep_dims)


def create_env(
    env_name,
    device,
    seed=None,
    num_envs=None,
    episode_length=1000,
    keep_dims_override=None,
):
    params = ENV_HYPERPARAMS[env_name]
    keep_dims = _validate_keep_dims(
        ENVS_DIM[env_name]["keep_dims"]
        if keep_dims_override is None
        else keep_dims_override
    )
    batch_size = params["num_envs"] if num_envs is None else int(num_envs)
    if batch_size < 1:
        raise ValueError("num_envs must be positive")
    base_env = envs.create(
        env_name,
        batch_size=batch_size,
        episode_length=_validate_episode_length(episode_length),
    )
    env = POMDPWrapper(base_env, keep_dims, seed=seed)
    env = torch_wrapper.TorchWrapper(env, device=device)
    return env
