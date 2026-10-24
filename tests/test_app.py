import app as webapp


def valid_form():
    return {
        "action": "score",
        "source": "Referral",
        "city": "Islamabad",
        "area": "",
        "property_type": "Apartment",
        "budget_pkr_lac": "100",
        "bedrooms": "2",
        "is_overseas": "0",
        "referred_by_existing_client": "0",
        "has_financing_approved": "1",
    }


class StubModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == [
            "source", "city", "area", "property_type", "budget_pkr_lac",
            "bedrooms", "is_overseas", "referred_by_existing_client", "has_financing_approved",
        ]
        return [[0.25, 0.75]]


def test_health_of_main_page_and_grounded_question_flow():
    client = webapp.app.test_client()

    assert client.get("/").status_code == 200
    response = client.post("/", data={"action": "ask", "question": "What is the transfer fee?"})
    assert response.status_code == 200
    assert b"There is a conflict" in response.data
    assert b"02_price_list_payment_plan.md" in response.data
    assert b"03_booking_policy_faq.md" in response.data


def test_invalid_lead_input_returns_helpful_client_error(monkeypatch):
    monkeypatch.setattr(webapp, "model", StubModel())
    data = valid_form()
    data["budget_pkr_lac"] = "NaN"

    response = webapp.app.test_client().post("/", data=data)

    assert response.status_code == 400
    assert b"must be at least 0" in response.data


def test_missing_model_returns_clear_service_unavailable(monkeypatch):
    monkeypatch.setattr(webapp, "model", None)

    response = webapp.app.test_client().post("/", data=valid_form())

    assert response.status_code == 503
    assert b"model has not been trained" in response.data


def test_valid_lead_input_returns_prediction(monkeypatch):
    monkeypatch.setattr(webapp, "model", StubModel())

    response = webapp.app.test_client().post("/", data=valid_form())

    assert response.status_code == 200
    assert b"75.0%" in response.data

