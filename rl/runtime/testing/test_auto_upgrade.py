"""Tests for headless-training auto-upgrade stat priorities."""

from __future__ import annotations

from rl.env.auto_upgrade import PRIMARY_COMBAT_STAT_ORDER, preset_auto_upgrade_policy


def _progression(*, legal_stats, current_tank=0):
    return {
        "current_tank": current_tank,
        "stat_levels": [0] * 8,
        "legal_stat_upgrades": list(legal_stats),
        "legal_tank_upgrades": [0] * 6,
    }


def test_primary_combat_stat_order_uses_headless_stat_slots():
    # Canonical headless combat slots:
    #   5 = BulletDamage, 6 = Reload, 4 = BulletPenetration,
    #   3 = BulletSpeed, 7 = MovementSpeed, 0 = MaxHealth.
    assert PRIMARY_COMBAT_STAT_ORDER == (5, 6, 4, 3, 7, 0)


def test_predator_policy_prefers_bullet_damage_then_reload_then_penetration():
    policy = preset_auto_upgrade_policy("predator")

    assert policy.stat_choice(_progression(legal_stats=[0, 0, 0, 0, 0, 1, 1, 1])) == 5
    assert policy.stat_choice(_progression(legal_stats=[0, 0, 0, 0, 1, 0, 1, 1])) == 6
    assert policy.stat_choice(_progression(legal_stats=[0, 0, 0, 0, 1, 0, 0, 1])) == 4
