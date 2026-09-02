-- 1) Conversion rate by lead source, best first, only sources with 200+ clean leads.
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

-- 2) Find duplicate leads in the original CRM dump.
-- We query the raw staging table because duplicates are intentionally rejected from
-- the clean operational table by crm_record_hash UNIQUE.
SELECT
    crm_record_hash,
    COUNT(*) AS duplicate_rows,
    MIN(created_at) AS first_created_at,
    MAX(created_at) AS last_created_at,
    GROUP_CONCAT(lead_id, ', ') AS lead_ids
FROM crm_import_raw
GROUP BY crm_record_hash
HAVING COUNT(*) > 1
ORDER BY duplicate_rows DESC, crm_record_hash;

-- Schema prevention:
-- leads.lead_id is the PRIMARY KEY, so IDs cannot repeat.
-- leads.crm_record_hash is UNIQUE, so the same CRM identity cannot be inserted twice.
-- Raw duplicates remain available in crm_import_raw for audit/reconciliation.
