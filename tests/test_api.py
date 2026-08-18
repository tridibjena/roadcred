"""Serving-layer tests.

These run without a trained model: the API must degrade cleanly when no ONNX artefact
exists rather than raising at import or returning a 500, because that is the state a fresh
clone is in.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving.api import _encode_png, _preprocess, _read_csv, _softmax, app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_always_responds(client):
    """Health must work whether or not a model is present."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "no_model"}
    assert "model" in body


def test_predict_without_model_returns_503_not_500(client):
    """A missing model is an expected state and must not look like a crash."""
    from serving.api import MODEL_CANDIDATES, REPO_ROOT

    # Check the filesystem, not bundle.session: the app loads lazily on first request,
    # so the session is still None here even when a model is available.
    if any((REPO_ROOT / "checkpoints" / n).exists() for n in MODEL_CANDIDATES):
        pytest.skip("a model is present; the no-model path cannot be exercised")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    response = client.post("/predict", files={"image": ("x.png", _png_bytes(image), "image/png")})
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()


def test_predict_rejects_undecodable_upload(client):
    from serving.api import MODEL_CANDIDATES, REPO_ROOT

    if not any((REPO_ROOT / "checkpoints" / n).exists() for n in MODEL_CANDIDATES):
        pytest.skip("no model present; the decode path is unreachable")
    response = client.post("/predict", files={"image": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_unknown_results_table_is_404(client):
    assert client.get("/results/definitely_not_a_table").status_code == 404


def test_figure_path_traversal_is_blocked(client):
    """A crafted name must not escape the figures directory."""
    for name in ("../../README.md", "..%2f..%2fREADME.md"):
        assert client.get(f"/figures/{name}").status_code in {404, 400}


def test_results_endpoint_shape(client):
    body = client.get("/results").json()
    assert set(body) >= {"tables", "figures"}
    assert isinstance(body["tables"], dict)


def test_softmax_normalises_and_matches_reference():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 7, 4, 5)).astype(np.float32)
    probs = _softmax(logits, axis=1)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert (probs >= 0).all()
    # Shift-invariance: softmax must be unaffected by a constant offset.
    np.testing.assert_allclose(probs, _softmax(logits + 12.0, axis=1), rtol=1e-5)


def test_preprocess_shape_and_normalisation():
    image = np.full((100, 200, 3), 128, dtype=np.uint8)
    out = _preprocess(image, (224, 320))
    assert out.shape == (1, 3, 224, 320)
    assert out.dtype == np.float32
    # 128/255 is near the ImageNet mean, so normalised values sit near zero.
    assert abs(float(out.mean())) < 1.0


def test_encode_png_roundtrips():
    from PIL import Image

    rng = np.random.default_rng(0)
    original = rng.integers(0, 255, (12, 15, 3), dtype=np.uint8)
    decoded = np.array(Image.open(io.BytesIO(base64.b64decode(_encode_png(original)))))
    np.testing.assert_array_equal(decoded, original)


def test_read_csv_coerces_numerics(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("name,miou,epochs,blank\nce,0.6182,30,\n")
    row = _read_csv(path)[0]
    assert row["name"] == "ce"
    assert isinstance(row["miou"], float) and row["miou"] == pytest.approx(0.6182)
    assert isinstance(row["epochs"], int) and row["epochs"] == 30
    assert row["blank"] is None


def _png_bytes(array: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
