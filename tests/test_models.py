import sys
import types

import pytest

from songtrace import models
from songtrace.stages import align, pitch


@pytest.fixture
def isolated_callbacks(monkeypatch):
    """release() with only the callbacks a test registers."""
    monkeypatch.setattr(models, "_callbacks", [])
    return models._callbacks


def fake_torch(available=True, initialised=True, emptied=None, reserved=None):
    """A torch whose allocator reports `reserved` bytes, then 0 once emptied."""
    state = {"reserved": 4 * 2 ** 20 if reserved is None else reserved}

    def empty_cache():
        emptied.append(True)
        state["reserved"] = 0

    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        is_initialized=lambda: initialised,
        memory_reserved=lambda: state["reserved"],
        empty_cache=empty_cache)
    return mod


def test_release_without_torch_or_callbacks_does_nothing(monkeypatch,
                                                         isolated_callbacks):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert models.release() == 0


def test_release_runs_every_registered_callback(isolated_callbacks):
    called = []
    models.on_release(lambda: called.append("a"))
    models.on_release(lambda: called.append("b"))
    models.release()
    assert called == ["a", "b"]


def test_on_release_returns_the_function_so_it_works_as_a_decorator():
    def drop():
        pass
    assert models.on_release(drop) is drop
    models._callbacks.remove(drop)


def test_release_empties_the_allocator_when_cuda_is_in_use(monkeypatch,
                                                           isolated_callbacks):
    emptied = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(emptied=emptied))
    assert models.release() == 4 * 2 ** 20
    assert emptied == [True]


def test_release_leaves_the_allocator_alone_without_a_gpu(monkeypatch,
                                                          isolated_callbacks):
    emptied = []
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(available=False, emptied=emptied))
    assert models.release() == 0
    assert emptied == []


def test_release_does_not_initialise_cuda_just_to_clean_up(monkeypatch,
                                                           isolated_callbacks):
    """A run that never touched the GPU must not create a context on the way out."""
    emptied = []
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(initialised=False, emptied=emptied))
    assert models.release() == 0
    assert emptied == []


def test_align_drops_its_cached_model():
    align._model_cache["cuda"] = object()
    align.unload_model()
    assert align._model_cache == {}


def test_align_registers_its_unload():
    assert align.unload_model in models._callbacks


def test_pitch_drops_the_model_torchcrepe_cached(monkeypatch):
    torchcrepe = types.ModuleType("torchcrepe")

    def infer():
        pass

    infer.model = object()
    infer.capacity = "full"
    torchcrepe.infer = infer
    monkeypatch.setitem(sys.modules, "torchcrepe", torchcrepe)
    pitch.unload_model()
    assert not hasattr(infer, "model")
    assert not hasattr(infer, "capacity")


def test_pitch_unload_is_fine_when_torchcrepe_was_never_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "torchcrepe", raising=False)
    pitch.unload_model()


def test_pitch_unload_is_fine_when_no_model_was_cached(monkeypatch):
    torchcrepe = types.ModuleType("torchcrepe")
    torchcrepe.infer = lambda: None
    monkeypatch.setitem(sys.modules, "torchcrepe", torchcrepe)
    pitch.unload_model()


def test_pitch_registers_its_unload():
    assert pitch.unload_model in models._callbacks


def test_release_reports_nothing_freed_when_the_allocator_was_already_empty(
        monkeypatch, isolated_callbacks):
    emptied = []
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(emptied=emptied, reserved=0))
    assert models.release() == 0
    assert emptied == [True]
