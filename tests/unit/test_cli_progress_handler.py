import pytest
from tqdm import tqdm

from comfy_execution.progress import CLIProgressHandler, NodeProgressState, NodeState


def _make_state(value: float = 0, max_: float = 10) -> NodeProgressState:
    return NodeProgressState(state=NodeState.Running, value=value, max=max_)


class _TqdmPatch:
    """Patches tqdm.__init__ and tqdm.update the same way comfy_tqdm() does."""

    def __enter__(self):
        self._orig_init = tqdm.__init__
        self._orig_update = tqdm.update

        orig_init = self._orig_init

        def patched_init(self_tqdm, *args, **kwargs):
            orig_init(self_tqdm, *args, **kwargs)
            self_tqdm._progress_bar = object()

        def patched_update(self_tqdm, n=1):
            assert self_tqdm._progress_bar is not None
            self._orig_update(self_tqdm, n)

        tqdm.__init__ = patched_init
        tqdm.update = patched_update
        return self

    def __exit__(self, *exc):
        tqdm.__init__ = self._orig_init
        tqdm.update = self._orig_update
        return False


class TestCLIProgressHandlerBypassesTqdmPatch:
    def test_bar_created_before_patch_updated_after(self):
        handler = CLIProgressHandler()
        state = _make_state()

        handler.start_handler("n1", state, "p1")
        bar = handler.progress_bars["n1"]
        assert not hasattr(bar, "_progress_bar")

        with _TqdmPatch():
            handler.update_handler("n1", 5, 10, state, "p1")
            assert bar.n == 5
            handler.finish_handler("n1", _make_state(10, 10), "p1")

        assert "n1" not in handler.progress_bars

    def test_bar_created_and_updated_during_patch(self):
        handler = CLIProgressHandler()
        state = _make_state()

        with _TqdmPatch():
            handler.start_handler("n1", state, "p1")
            bar = handler.progress_bars["n1"]
            assert not hasattr(bar, "_progress_bar")

            handler.update_handler("n1", 7, 10, state, "p1")
            assert bar.n == 7
            handler.finish_handler("n1", _make_state(10, 10), "p1")

        assert "n1" not in handler.progress_bars

    def test_update_creates_bar_on_demand(self):
        handler = CLIProgressHandler()
        state = _make_state()

        with _TqdmPatch():
            handler.update_handler("n1", 3, 10, state, "p1")
            bar = handler.progress_bars["n1"]
            assert not hasattr(bar, "_progress_bar")
            assert bar.n == 3
            assert bar.total == 10

            handler.finish_handler("n1", _make_state(10, 10), "p1")

    def test_total_changes_mid_progress(self):
        handler = CLIProgressHandler()
        state = _make_state()

        handler.start_handler("n1", state, "p1")
        bar = handler.progress_bars["n1"]
        assert bar.total == 10

        handler.update_handler("n1", 5, 20, state, "p1")
        assert bar.total == 20
        assert bar.n == 5

        handler.finish_handler("n1", _make_state(20, 20), "p1")

    def test_reset_closes_all_bars(self):
        handler = CLIProgressHandler()
        state = _make_state()

        with _TqdmPatch():
            handler.start_handler("n1", state, "p1")
            handler.start_handler("n2", state, "p2")
            assert len(handler.progress_bars) == 2

            handler.reset()
            assert len(handler.progress_bars) == 0

    def test_no_infinite_recursion_via_patched_update(self):
        handler = CLIProgressHandler()
        state = _make_state()

        orig_update = tqdm.update
        calls_to_patched = []

        def spy_update(self_tqdm, n=1):
            calls_to_patched.append(id(self_tqdm))
            orig_update(self_tqdm, n)

        tqdm.update = spy_update
        try:
            handler.start_handler("n1", state, "p1")
            bar = handler.progress_bars["n1"]
            bar_id = id(bar)

            handler.update_handler("n1", 5, 10, state, "p1")
            handler.finish_handler("n1", _make_state(10, 10), "p1")

            assert bar_id not in calls_to_patched
        finally:
            tqdm.update = orig_update

    def test_class_attrs_are_not_patched_versions(self):
        bar = tqdm.__new__(tqdm)
        CLIProgressHandler._tqdm_init(bar, total=1, disable=True)
        assert not hasattr(bar, "_progress_bar")
        CLIProgressHandler._tqdm_update(bar, 1)
        bar.close()
