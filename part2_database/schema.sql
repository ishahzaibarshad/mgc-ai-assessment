-- Minimal normalized-enough CRM table for this assessment.
-- The raw CSV should be cleaned before loading (e.g. city spelling normalization).
CREATE TABLE leads (
    lead_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    source TEXT NOT NULL,
    city TEXT NOT NULL,
    area TEXT,
    property_type TEXT NOT NULL,
    budget_pkr_lac DECIMAL(12,2),
    bedrooms INTEGER,
    first_response_minutes INTEGER,
    calls_made INTEGER NOT NULL DEFAULT 0,
    total_call_seconds INTEGER NOT NULL DEFAULT 0,
    whatsapp_replies INTEGER NOT NULL DEFAULT 0,
    site_visits INTEGER NOT NULL DEFAULT 0,
    agent_experience_years DECIMAL(4,1),
    is_overseas INTEGER NOT NULL DEFAULT 0,
    referred_by_existing_client INTEGER NOT NULL DEFAULT 0,
    has_financing_approved INTEGER NOT NULL DEFAULT 0,
    token_amount_received_pkr DECIMAL(15,2) NOT NULL DEFAULT 0,
    crm_record_hash BIGINT NOT NULL UNIQUE,
    converted INTEGER NOT NULL DEFAULT 0
);

-- crm_record_hash is unique because the supplied dump contains repeated hashes
-- for the same underlying CRM lead. This prevents the same lead from being inserted twice.
-- In production, keep the raw import table separate so duplicates can be audited before
-- loading into the clean leads table.
