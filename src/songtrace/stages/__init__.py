from ..pipeline import Stage
from .acquire import acquire_stage
from .align import align_stage
from .grid import grid_stage
from .notes import notes_stage
from .package import package_stage
from .pitch import pitch_stage
from .separate import separate_stage
from .transcribe import transcribe_stage


def all_stages() -> list[Stage]:
    return [acquire_stage(), separate_stage(), transcribe_stage(), align_stage(),
            pitch_stage(), notes_stage(), grid_stage(), package_stage()]
