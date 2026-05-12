from __future__ import annotations

import argparse
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Set reproducible random seeds for local experiment scripts.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(seed_everything(args.seed))


if __name__ == "__main__":
    main()
