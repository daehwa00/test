import hashlib
import json
import random
import warnings

import numpy as np
import torch


RNG_SCHEME = "time-rng-v1"
RNG_PROTOCOL = "time-brax-corrected-v2"


torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=FutureWarning)


def derive_seed(env_name: str, training_seed: int, domain: str, index: int = 0) -> int:
    """Derive a stable 32-bit seed without coupling consumers' draw counts."""
    payload = {
        "protocol": RNG_PROTOCOL,
        "task": env_name,
        "training_seed": int(training_seed),
        "domain": domain,
        "index": int(index),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(("TiME-RNG-v1\0" + canonical).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def set_seed(seed: int) -> None:
    """Seed process-local libraries; hash seeding must happen before startup."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
