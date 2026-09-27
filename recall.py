"""Stable integration interface for the gameplay teammate.

Call recall once per episode, then cursor.next_action before *each* move.
This local function does not pretend to have contacted either sponsor.
"""

from memory.core import MemoryStore, ProcedureCursor
from memory.service import recall_procedure


def recall(layout, observation=None, memory_dir="data/memory"):
    procedure = MemoryStore(memory_dir).recall(layout, observation)
    return ProcedureCursor(procedure) if procedure else None


def recall_from_gbrain(layout, observation=None, memory_dir="data/memory", gbrain=None):
    """Real GBrain retrieval. Raises GBrainError on outage; caller may label fallback."""
    result = recall_procedure(layout, observation, MemoryStore(memory_dir), "gbrain", gbrain)
    procedure = result["procedure"]
    return ProcedureCursor(procedure) if procedure else None
