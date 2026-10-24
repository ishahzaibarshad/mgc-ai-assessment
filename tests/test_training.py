import pandas as pd
import pytest

from part3_ml.train import FEATURES, train_and_evaluate


def sample_leads(rows=40):
    return pd.DataFrame(
        {
            "source": ["Referral" if i % 2 else "Online" for i in range(rows)],
            "city": ["ISB" if i % 3 else "RWP" for i in range(rows)],
            "area": [None if i % 5 == 0 else f"Area {i % 4}" for i in range(rows)],
            "property_type": ["Apartment" if i % 2 else "House" for i in range(rows)],
            "budget_pkr_lac": [50 + i * 2 for i in range(rows)],
            "bedrooms": [None if i % 7 == 0 else 1 + i % 4 for i in range(rows)],
            "is_overseas": [i % 2 for i in range(rows)],
            "referred_by_existing_client": [(i // 2) % 2 for i in range(rows)],
            "has_financing_approved": [(i // 3) % 2 for i in range(rows)],
            "converted": [i % 2 for i in range(rows)],
            # These post-contact fields must never be passed to the model.
            "calls_made": [1000] * rows,
            "first_response_minutes": [1] * rows,
            "token_amount_received_pkr": [999999] * rows,
        }
    )


def test_training_uses_only_pre_call_features_and_saves_requested_model(tmp_path):
    model_path = tmp_path / "baseline.joblib"

    result = train_and_evaluate(sample_leads(), model_path=model_path)

    assert result["rows"] == 40
    assert result["test_rows"] == 8
    assert 0 <= result["average_precision"] <= 1
    assert model_path.exists()
    assert list(result["model"].feature_names_in_) == FEATURES
    assert not {"calls_made", "first_response_minutes", "token_amount_received_pkr"}.intersection(
        result["model"].feature_names_in_
    )


def test_training_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        train_and_evaluate(pd.DataFrame({"converted": [0, 1]}))


def test_training_rejects_single_class_data():
    data = sample_leads()
    data["converted"] = 1

    with pytest.raises(ValueError, match="at least two examples from each target class"):
        train_and_evaluate(data)

