from ..pipeline import Stage
from .acquire import acquire_stage


def all_stages() -> list[Stage]:
    return [acquire_stage()]
