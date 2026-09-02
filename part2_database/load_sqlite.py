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


def main():
    df = pd.read_csv(CSV_PATH)
    df["city"] = df["city"].apply(normalize_city)

    # Keep the first occurrence of each CRM identity in the clean operational table.
    # The raw table is retained so duplicate records remain auditable.
    duplicate_rows = int(df.duplicated("crm_record_hash", keep=False).sum())
    duplicate_groups = int(df.loc[df.duplicated("crm_record_hash", keep=False), "crm_record_hash"].nunique())
    clean_df = df.drop_duplicates("crm_record_hash", keep="first").copy()

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DROP TABLE IF EXISTS crm_import_raw")
        conn.execute("DROP TABLE IF EXISTS leads")

        # Raw staging table: preserves the CRM dump exactly enough for duplicate auditing.
        df.to_sql("crm_import_raw", conn, if_exists="replace", index=False)

        # Clean operational table: use the submitted schema so PRIMARY KEY / UNIQUE
        # constraints are actually enforced rather than merely documented.
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        columns = [row[1] for row in conn.execute("PRAGMA table_info(leads)").fetchall()]
        clean_df[columns].to_sql("leads", conn, if_exists="append", index=False)
        conn.commit()

        print(f"Raw CRM rows: {len(df)}")
        print(f"Duplicate hash groups: {duplicate_groups}")
        print(f"Rows affected by duplicate hashes: {duplicate_rows}")
        print(f"Clean leads loaded: {len(clean_df)}")

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

        print("\nDuplicate CRM hashes in raw import:")
        duplicate_sql = """
        SELECT
            crm_record_hash,
            COUNT(*) AS duplicate_rows,
            MIN(created_at) AS first_created_at,
            MAX(created_at) AS last_created_at,
            GROUP_CONCAT(lead_id, ', ') AS lead_ids
        FROM crm_import_raw
        GROUP BY crm_record_hash
        HAVING COUNT(*) > 1
        ORDER BY duplicate_rows DESC, crm_record_hash
        LIMIT 10;
        """
        print(pd.read_sql_query(duplicate_sql, conn).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
