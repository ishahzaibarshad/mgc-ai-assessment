import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "part2_database" / "mgc.db"
SCHEMA_PATH = ROOT / "part2_database" / "schema.sql"
CSV_PATH = ROOT / "data" / "leads.csv"


def normalize_city(value):
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    mapping = {
        "ISLAMABAD": "Islamabad",
        "ISB": "Islamabad",
        "RAWALPINDI": "Rawalpindi",
        "RWP": "Rawalpindi",
        "LAHORE": "Lahore",
        "KARACHI": "Karachi",
        "KHI": "Karachi",
        "PESHAWAR": "Peshawar",
        "FAISALABAD": "Faisalabad",
        "MULTAN": "Multan",
        "GUJRANWALA": "Gujranwala",
    }
    return mapping.get(value, str(value).title())


def prepare_data(csv_path=CSV_PATH):
    """Load, normalize, and deduplicate a CRM export without mutating the input file."""
    df = pd.read_csv(csv_path)
    required = {"city", "crm_record_hash"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CRM export is missing required columns: {', '.join(sorted(missing))}")

    df["city"] = df["city"].apply(normalize_city)
    duplicate_mask = df.duplicated("crm_record_hash", keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_groups = int(df.loc[duplicate_mask, "crm_record_hash"].nunique())
    clean_df = df.drop_duplicates("crm_record_hash", keep="first").copy()
    summary = {
        "raw_rows": len(df),
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "clean_rows": len(clean_df),
    }
    return df, clean_df, summary


def write_database(raw_df, clean_df, db_path=DB_PATH):
    """Build a replacement database beside the current one, then atomically swap it in."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = db_path.with_name(f".{db_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    conn = None
    try:
        conn = sqlite3.connect(temporary_path)
        conn.execute("PRAGMA foreign_keys = ON")
        raw_df.to_sql("crm_import_raw", conn, if_exists="replace", index=False)
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        columns = [row[1] for row in conn.execute("PRAGMA table_info(leads)").fetchall()]
        missing = set(columns).difference(clean_df.columns)
        if missing:
            raise ValueError(f"CRM export is missing schema columns: {', '.join(sorted(missing))}")
        clean_df[columns].to_sql("leads", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()
        conn = None
        os.replace(temporary_path, db_path)
    except Exception:
        if conn is not None:
            conn.rollback()
            conn.close()
        temporary_path.unlink(missing_ok=True)
        raise


def main():
    raw_df, clean_df, summary = prepare_data()
    write_database(raw_df, clean_df)

    print(f"Raw CRM rows: {summary['raw_rows']}")
    print(f"Duplicate hash groups: {summary['duplicate_groups']}")
    print(f"Rows affected by duplicate hashes: {summary['duplicate_rows']}")
    print(f"Clean leads loaded: {summary['clean_rows']}")
    print("Duplicate identifiers are omitted from console output.")

    with sqlite3.connect(DB_PATH) as conn:
        print("\nConversion rates on clean leads:")
        conversion_sql = """
        SELECT
            source,
            COUNT(*) AS lead_count,
            SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) AS converted_count,
            ROUND(
                100.0 * SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) / COUNT(*),
                2
            ) AS conversion_rate_pct
        FROM leads
        GROUP BY source
        HAVING COUNT(*) >= 200
        ORDER BY conversion_rate_pct DESC;
        """
        print(pd.read_sql_query(conversion_sql, conn).to_string(index=False))


if __name__ == "__main__":
    main()

