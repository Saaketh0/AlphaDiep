"""Shared policy IDs for main and ghost league policies."""

CHAR_CLASSES = ("A", "B", "C", "D")
GHOST_SLOTS = 4

MAIN_POLICIES = [f"main_class_{char_class}" for char_class in CHAR_CLASSES]
GHOST_POLICIES = [
    f"class_{char_class}_ghost_{slot}"
    for char_class in CHAR_CLASSES
    for slot in range(GHOST_SLOTS)
]


def main_policy_id(char_class: str) -> str:
    return f"main_class_{char_class}"


def ghost_policy_id(char_class: str, slot: int) -> str:
    return f"class_{char_class}_ghost_{int(slot)}"


def agent_index_from_id(agent_id: str) -> int:
    """Parse the numeric suffix from a PettingZoo agent name (``agent_7`` → 7)."""
    return int(agent_id.split("_")[-1])


def policy_id_for_agent(agent_id: str) -> str:
    """Map a PettingZoo agent ID to its RLlib policy ID (main or ghost)."""
    index = agent_index_from_id(agent_id)
    if index < len(CHAR_CLASSES):
        return MAIN_POLICIES[index]
    char_class = CHAR_CLASSES[index % len(CHAR_CLASSES)]
    ghost_slot = (index // len(CHAR_CLASSES)) % GHOST_SLOTS
    return ghost_policy_id(char_class, ghost_slot)
