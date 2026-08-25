from __future__ import annotations

from typing import Any

import torch


OPTIMIZER_GROUP_ID = "optimizer_group_id"
SCHEDULE_GROUP = "schedule_group"


def register_stable_optimizer_groups(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    _assign_group_identities(optimizer.param_groups)
    identities = _group_identities(optimizer.param_groups)
    if len(identities) != len(set(identities)):
        raise ValueError("optimizer group identities must be unique")

    def align_groups(
        current: torch.optim.Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        saved_groups = state_dict.get("param_groups")
        if not isinstance(saved_groups, list):
            return state_dict
        saved_identities = [group.get(OPTIMIZER_GROUP_ID) for group in saved_groups]
        if not any(saved_identities):
            return state_dict
        if any(not isinstance(identity, str) for identity in saved_identities):
            raise ValueError("checkpoint optimizer groups have incomplete identities")
        if len(saved_identities) != len(set(saved_identities)):
            raise ValueError("checkpoint optimizer group identities must be unique")
        current_identities = _group_identities(current.param_groups)
        if set(saved_identities) != set(current_identities):
            raise ValueError("checkpoint optimizer group identities do not match")
        by_identity = dict(zip(saved_identities, saved_groups, strict=True))
        state_dict["param_groups"] = [
            by_identity[identity] for identity in current_identities
        ]
        return state_dict

    optimizer.register_load_state_dict_pre_hook(align_groups)
    return optimizer


def _group_identities(groups: list[dict[str, Any]]) -> list[str]:
    identities = [group.get(OPTIMIZER_GROUP_ID) for group in groups]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("every optimizer group needs an optimizer_group_id")
    return identities


def _assign_group_identities(groups: list[dict[str, Any]]) -> None:
    existing = [group.get(OPTIMIZER_GROUP_ID) for group in groups]
    if any(existing):
        if any(not isinstance(identity, str) or not identity for identity in existing):
            raise ValueError("optimizer groups have incomplete optimizer_group_id values")
        return
    role_counts: dict[str, int] = {}
    for group in groups:
        role = group.get(SCHEDULE_GROUP)
        if not isinstance(role, str) or not role:
            raise ValueError("every optimizer group needs a schedule_group role")
        role_index = role_counts.get(role, 0)
        group[OPTIMIZER_GROUP_ID] = f"{role}:{role_index}"
        role_counts[role] = role_index + 1
