import sqlite3

import pandas as pd
import pytest

from part2_database.load_sqlite import normalize_city, prepare_data, write_database


def record(lead_id, crm_hash, city):
    return {
        "lead_id": lead_id,
        "created_at": "2025-01-01 12:00:00",
        "source": "Referral",
        "city": city,
        "area": None,
        "property_type": "Apartment",
        "budget_pkr_lac": 100.0,
        "bedrooms": 2,
        "first_response_minutes": 10,
        "calls_made": 0,
        "total_call_seconds": 0,
        "whatsapp_replies": 0,
        "site_visits": 0,
        "agent_experience_years": 1.0,
        "is_overseas": 0,
        "referred_by_existing_client": 0,
        "has_financing_approved": 0,
        "token_amount_received_pkr": 0,
        "crm_record_hash": crm_hash,
        "converted": 0,
    }


def test_city_normalization_handles_aliases_unknown_values_and_missing_values():
    assert normalize_city(" ISB ") == "Islamabad"
    assert normalize_city("new city") == "New City"
    assert normalize_city(None) is None


def test_prepare_data_deduplicates_crm_hashes_and_counts_affected_rows(tmp_path):
    csv_path = tmp_path / "leads.csv"
    pd.DataFrame(
        [record("A", 1001, "ISB"), record("A-copy", 1001, "Islamabad"), record("B", 1002, "RWP")]
    ).to_csv(csv_path, index=False)

    raw, clean, summary = prepare_data(csv_path)

    assert len(raw) == 3
    assert len(clean) == 2
    assert summary == {"raw_rows": 3, "duplicate_groups": 1, "duplicate_rows": 2, "clean_rows": 2}
    assert clean.iloc[0]["city"] == "Islamabad"
    assert clean.iloc[1]["city"] == "Rawalpindi"


def test_database_loader_keeps_raw_rows_and_inserts_deduplicated_rows(tmp_path):
    csv_path = tmp_path / "leads.csv"
    pd.DataFrame([record("A", 1001, "ISB"), record("A-copy", 1001, "Islamabad")]).to_csv(csv_path, index=False)
    raw, clean, _ = prepare_data(csv_path)
    database_path = tmp_path / "mgc.db"

    write_database(raw, clean, database_path)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM crm_import_raw").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0] == 1
        assert connection.execute("SELECT city FROM leads").fetchone()[0] == "Islamabad"


def test_failed_reload_preserves_existing_database(tmp_path):
    database_path = tmp_path / "mgc.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE keep_me (value TEXT NOT NULL)")
        connection.execute("INSERT INTO keep_me VALUES ('existing data')")

    incomplete = pd.DataFrame([{"city": "ISB", "crm_record_hash": 1001}])
    with pytest.raises(ValueError, match="missing schema columns"):
        write_database(incomplete, incomplete, database_path)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT value FROM keep_me").fetchone()[0] == "existing data"
    assert not database_path.with_name(f".{database_path.name}.tmp").exists()

