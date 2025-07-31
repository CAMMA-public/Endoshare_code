from enum import Enum


class ProcessingMode(Enum):
    NORMAL = 0
    ADVANCED = 1


class ProcessingInterrupted(Exception):
    """Raised to abort processing when user hits Terminate."""
    pass
