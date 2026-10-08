"""Legacy entrypoint for controlled oracle sanity checks.

Prefer ``experiments/run_sanity_check.py`` for Phase 3 experiments.
"""

from __future__ import annotations

from experiments.run_sanity_check import main

if __name__ == "__main__":
    main()
