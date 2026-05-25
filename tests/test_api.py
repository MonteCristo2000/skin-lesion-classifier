from serving.app import CLASS_NAMES, MODEL_VERSION


# --- /health ---

def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_schema(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == MODEL_VERSION


# --- /predict happy path ---

def test_predict_jpeg_returns_200(client, jpeg_bytes):
    r = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200


def test_predict_png_returns_200(client, png_bytes):
    r = client.post("/predict", files={"file": ("lesion.png", png_bytes, "image/png")})
    assert r.status_code == 200


def test_predict_response_has_required_fields(client, jpeg_bytes):
    body = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert "predicted_class" in body
    assert "confidence" in body
    assert "probabilities" in body
    assert "model_version" in body


def test_predict_model_version(client, jpeg_bytes):
    body = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert body["model_version"] == MODEL_VERSION


def test_predict_all_classes_in_probabilities(client, jpeg_bytes):
    body = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert set(body["probabilities"].keys()) == set(CLASS_NAMES)


def test_predict_probabilities_sum_to_one(client, jpeg_bytes):
    body = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4


def test_predict_confidence_matches_predicted_class(client, jpeg_bytes):
    body = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert body["confidence"] == body["probabilities"][body["predicted_class"]]


def test_predict_confidence_in_range(client, jpeg_bytes):
    body = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_predicted_class_is_valid(client, jpeg_bytes):
    body = client.post("/predict", files={"file": ("lesion.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert body["predicted_class"] in CLASS_NAMES


# --- /predict error cases ---

def test_predict_pdf_returns_400(client):
    r = client.post("/predict", files={"file": ("report.pdf", b"not-an-image", "application/pdf")})
    assert r.status_code == 400


def test_predict_pdf_error_message(client):
    r = client.post("/predict", files={"file": ("report.pdf", b"not-an-image", "application/pdf")})
    detail = r.json()["detail"]
    assert "JPEG" in detail or "PNG" in detail
