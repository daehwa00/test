import utils
from trainer import train
import torch
from env_utils import ENV_HYPERPARAMS
import argparse

SUPPORTED_ENVS = list(ENV_HYPERPARAMS.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="halfcheetah",
        choices=SUPPORTED_ENVS,
        help=f"Environment name. Supported: {', '.join(SUPPORTED_ENVS)}",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--ef-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Episodic Forgetting (default: enabled)",
    )
    parser.add_argument(
        "--mr-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Memory Rebalancer (default: enabled)",
    )
    parser.add_argument(
        "--reset-dt",
        type=float,
        default=5.0,
        help="Episode-boundary reset delta used by EF (default: 5.0)",
    )

    args = parser.parse_args()

    config = {
        **ENV_HYPERPARAMS[args.env],
        "epsilon": 0.2,
        "num_minibatches": 4,
        "num_update_epochs": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "env_name": args.env,
        "use_wandb": args.use_wandb,
        "ef_enabled": args.ef_enabled,
        "mr_enabled": args.mr_enabled,
        "reset_dt": args.reset_dt,
    }

    print(f"Training for environment: {args.env}, args.seed: {args.seed}")
    train(config=config, env_name=args.env)


if __name__ == "__main__":
    main()
