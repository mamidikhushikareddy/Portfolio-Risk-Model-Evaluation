"""Headless Streamlit tests (no browser, no network)."""
import pytest

pytest.importorskip("streamlit")
pytestmark = pytest.mark.app
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _synthetic_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("Synthetic demo").run()
    return at


def test_every_page_renders_and_synthetic_is_bannered():
    at = _synthetic_app()
    assert any("SYNTHETIC" in e.value for e in at.error)
    for page in at.sidebar.radio[-1].options:
        at.sidebar.radio[-1].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])


def test_invalid_weights_block_metrics_and_normalise_recovers():
    at = _synthetic_app()
    at.sidebar.number_input[2].set_value(35.0).run()  # first weight input after risk-free and window
    assert any("must sum to exactly 100%" in e.value for e in at.error)
    assert not at.metric  # no metrics computed on invalid weights
    next(b for b in at.sidebar.button if "Normalise" in b.label).click().run()
    assert not at.exception and not any("must sum" in e.value for e in at.error)
    assert at.metric  # overview metrics now present


def test_weight_change_recalculates_metrics():
    at = _synthetic_app()
    vol_before = [m.value for m in at.metric][1]
    for i, w in zip(range(2, 9), [0, 0, 100, 0, 0, 0, 0], strict=True):  # 100% AGG
        at.sidebar.number_input[i].set_value(float(w))
    at.run()
    assert [m.value for m in at.metric][1] != vol_before
