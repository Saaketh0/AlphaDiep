"""Tests for shared league constants and policy mapping."""

from __future__ import annotations

import pytest

from league_initialization.constants import (
    CHAR_CLASSES,
    GHOST_SLOTS,
    MAIN_POLICIES,
    ghost_policy_id,
    main_policy_id,
    policy_id_for_agent,
)


@pytest.mark.parametrize(
    ("agent_id", "expected"),
    [
        ("agent_0", main_policy_id("A")),
        ("agent_3", main_policy_id("D")),
        ("agent_4", ghost_policy_id("A", 1)),
        ("agent_7", ghost_policy_id("D", 1)),
        ("agent_8", ghost_policy_id("A", 2)),
        ("agent_19", ghost_policy_id("D", 0)),
    ],
)
def test_policy_id_for_agent(agent_id, expected):
    assert policy_id_for_agent(agent_id) == expected


def test_policy_id_for_agent_covers_all_twenty_agents():
    mains = {policy_id_for_agent(f"agent_{index}") for index in range(len(CHAR_CLASSES))}
    assert mains == set(MAIN_POLICIES)

    ghosts = {
        policy_id_for_agent(f"agent_{index}")
        for index in range(len(CHAR_CLASSES), len(CHAR_CLASSES) * (1 + GHOST_SLOTS))
    }
    expected_ghosts = {
        ghost_policy_id(char_class, slot)
        for char_class in CHAR_CLASSES
        for slot in range(GHOST_SLOTS)
    }
    assert ghosts == expected_ghosts
