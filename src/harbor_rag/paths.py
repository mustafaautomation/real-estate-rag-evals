from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Find the checkout that holds evals/config.yaml.

    Resolution order is HARBOR_ROOT, then the current working directory and
    its parents, then this file's parents. Commands are meant to be run from
    the repository.
    """

    env = os.environ.get("HARBOR_ROOT")
    if env:
        return Path(env)
    starts = [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "evals" / "config.yaml").is_file():
                return candidate
    raise FileNotFoundError("Could not find evals/config.yaml. Set HARBOR_ROOT to the repo.")
