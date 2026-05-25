from serving.schemas import HealthResponse, PredictionResponse


def test_health_response():
    r = HealthResponse(status="ok", model_loaded=True, model_version="1.0.0")
    assert r.status == "ok"
    assert r.model_loaded is True


def test_prediction_response():
    r = PredictionResponse(
        predicted_class="Melanoma",
        confidence=0.92,
        probabilities={"Melanoma": 0.92, "Nevus (mole)": 0.08},
        model_version="1.0.0",
    )
    assert r.predicted_class == "Melanoma"
    assert r.confidence == 0.92
