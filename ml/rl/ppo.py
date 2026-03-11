from __future__ import annotations

from .common import RLConfig, run_sb3_workflow


def run_ppo_workflow(*, bt_panel, cfg: RLConfig, train_split_date, years):
    return run_sb3_workflow(
        bt_panel=bt_panel,
        cfg=cfg,
        train_split_date=train_split_date,
        years=years,
        algorithm="ppo",
    )


__all__ = ["RLConfig", "run_ppo_workflow"]
