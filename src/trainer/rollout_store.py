"""Utilities for saving / loading rollouts as JSON and rendering them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

__all__ = ["RolloutStore"]


class RolloutStore:
    """Save rollouts in *one JSON file per rollout* fashion."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def clear_all_rollouts(self):
        """Delete all JSON files in the rollouts directory."""
        for json_file in self.dir.glob("*.json"):
            json_file.unlink()
        self._counter = 0

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------

    def save(self, rollout: Dict):
        path = self.dir / f"rollout_{self._counter:08d}.json"
        with path.open("w") as fp:
            json.dump(rollout, fp, indent=2)
        self._counter += 1

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def render(self, k: int = 1, output_file: str | None = None):
        """Render every *k*-th rollout to a pretty text table."""
        files = sorted(self.dir.glob("rollout_*.json"))[::k]
        console = Console(record=True)

        for file in files:
            with file.open() as fp:
                data = json.load(fp)
            table = Table(title=str(file.name))
            # add summary row
            rb = data.get("reward_breakdown", {})
            summary = {
                "total_reward": data.get("total_reward"),
                **{k: f"{v:.3f}" for k, v in rb.items()},
            }
            for key, val in summary.items():
                table.add_row(key, str(val))
            table.add_row("prompt", data.get("prompt", "")[:80] + "…")
            table.add_row("response", data.get("response", "")[:80] + "…")
            console.print(table)

        if output_file is not None:
            console.save_text(output_file)
