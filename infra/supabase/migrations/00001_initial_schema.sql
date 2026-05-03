-- Roof Corrosion MLOps — Initial Schema
-- Supabase Postgres migrations

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Customers ──────────────────────────────────────────────
CREATE TABLE customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auth_user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    company_name TEXT,
    phone TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(email)
);

-- ── Jobs (quote requests) ──────────────────────────────────
CREATE TYPE job_status AS ENUM ('queued', 'processing', 'completed', 'failed', 'requires_review');
CREATE TYPE job_source AS ENUM ('maxar', 'nearmap', 'drone', 'uploaded');

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    status job_status NOT NULL DEFAULT 'queued',
    source job_source NOT NULL DEFAULT 'maxar',

    -- Location
    address TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    aoi_geojson JSONB,  -- custom area of interest polygon

    -- Tile metadata
    tile_zoom INT DEFAULT 20,
    tile_capture_date DATE,
    gsd_m DOUBLE PRECISION,  -- ground sample distance in meters

    -- Processing
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    processing_time_ms INT,

    -- Model provenance (audit trail)
    roof_model_version TEXT,
    corrosion_model_version TEXT,
    mlflow_run_id TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_jobs_customer ON jobs(customer_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_submitted ON jobs(submitted_at DESC);

-- ── Corrosion assessments ─────────────────────────────────
CREATE TYPE corrosion_severity AS ENUM ('none', 'light', 'moderate', 'severe');

CREATE TABLE assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE UNIQUE,

    -- Measurements
    roof_area_m2 DOUBLE PRECISION NOT NULL,
    corroded_area_m2 DOUBLE PRECISION NOT NULL,
    corrosion_percent DOUBLE PRECISION NOT NULL,
    severity corrosion_severity NOT NULL DEFAULT 'none',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- 0–1

    -- Masks stored as S3 references (not in DB)
    roof_mask_s3_key TEXT,
    corrosion_mask_s3_key TEXT,
    overlay_image_s3_key TEXT,

    -- Uncertainty map (for active learning sampling)
    uncertainty_map_s3_key TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Quotes ──────────────────────────────────────────────────
CREATE TABLE quotes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    assessment_id UUID NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,

    -- Pricing
    currency TEXT NOT NULL DEFAULT 'USD',
    total_amount DOUBLE PRECISION NOT NULL,
    line_items JSONB NOT NULL,  -- [{description, quantity, unit, unit_price, total}]

    -- Flags
    requires_human_review BOOLEAN NOT NULL DEFAULT false,
    reviewed_by UUID REFERENCES customers(id),
    reviewed_at TIMESTAMPTZ,
    approved BOOLEAN,

    -- PDF
    pdf_s3_key TEXT,

    -- Validity
    valid_until TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_quotes_job ON quotes(job_id);

-- ── Feedback (for model improvement loop) ──────────────────
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,

    correct BOOLEAN NOT NULL,
    notes TEXT,
    flagged_for_relabeling BOOLEAN NOT NULL DEFAULT false,

    -- Which aspect was wrong?
    roof_boundary_wrong BOOLEAN DEFAULT false,
    corrosion_area_wrong BOOLEAN DEFAULT false,
    severity_wrong BOOLEAN DEFAULT false,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_feedback_relabeling ON feedback(flagged_for_relabeling) WHERE flagged_for_relabeling = true;

-- ── Price book (configurable per material/region) ──────────
CREATE TABLE price_book (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    material TEXT NOT NULL,           -- e.g. 'corrugated_metal', 'tile', 'asphalt_shingle'
    service_type TEXT NOT NULL,       -- e.g. 'replacement', 'repair', 'coating'
    region TEXT NOT NULL DEFAULT 'default',
    price_per_m2 DOUBLE PRECISION NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(material, service_type, region, effective_from)
);

-- Seed price book with Thailand (THB) defaults — primary market.
-- Prices are representative THB/m² for Thai suppliers (BlueScope, SCG, local steel).
INSERT INTO price_book (material, service_type, region, price_per_m2, currency) VALUES
    -- Corrugated galvanized zinc/steel — the dominant Thai residential/industrial roof
    ('corrugated_metal', 'replacement', 'TH', 850.00, 'THB'),
    ('corrugated_metal', 'repair',      'TH', 450.00, 'THB'),
    ('corrugated_metal', 'coating',     'TH', 280.00, 'THB'),
    -- Ceramic / terracotta tile — traditional Thai roofs
    ('tile',             'replacement', 'TH', 1200.00, 'THB'),
    ('tile',             'repair',      'TH',  650.00, 'THB'),
    -- Keep global defaults for non-TH markets (USD)
    ('corrugated_metal', 'replacement', 'default', 45.00, 'USD'),
    ('corrugated_metal', 'repair',      'default', 25.00, 'USD'),
    ('corrugated_metal', 'coating',     'default', 15.00, 'USD'),
    ('tile',             'replacement', 'default', 65.00, 'USD'),
    ('tile',             'repair',      'default', 35.00, 'USD'),
    ('asphalt_shingle',  'replacement', 'default', 55.00, 'USD'),
    ('asphalt_shingle',  'repair',      'default', 30.00, 'USD');

-- ── Model registry (audit trail) ──────────────────────────
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name TEXT NOT NULL,         -- 'roof_detector' or 'corrosion_detector'
    version TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'dev', -- dev, staging, production
    mlflow_run_id TEXT,
    data_sources TEXT[] NOT NULL,     -- ['caribbean', 'airs', 'spacenet']
    metrics JSONB NOT NULL,           -- {roof_iou: 0.87, corrosion_iou: 0.52, ...}
    frozen_eval_iou DOUBLE PRECISION,
    frozen_eval_mape DOUBLE PRECISION,
    promoted_at TIMESTAMPTZ,
    promoted_by UUID REFERENCES customers(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(model_name, version)
);

-- ── RLS Policies ───────────────────────────────────────────
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE quotes ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- Customers can only see their own data
CREATE POLICY customers_own_data ON customers
    FOR ALL USING (auth_user_id = auth.uid());

CREATE POLICY jobs_own_data ON jobs
    FOR ALL USING (customer_id IN (
        SELECT id FROM customers WHERE auth_user_id = auth.uid()
    ));

CREATE POLICY assessments_via_job ON assessments
    FOR ALL USING (job_id IN (
        SELECT j.id FROM jobs j
        JOIN customers c ON j.customer_id = c.id
        WHERE c.auth_user_id = auth.uid()
    ));

CREATE POLICY quotes_via_job ON quotes
    FOR ALL USING (job_id IN (
        SELECT j.id FROM jobs j
        JOIN customers c ON j.customer_id = c.id
        WHERE c.auth_user_id = auth.uid()
    ));

CREATE POLICY feedback_own ON feedback
    FOR ALL USING (customer_id IN (
        SELECT id FROM customers WHERE auth_user_id = auth.uid()
    ));

-- Internal ops users can see everything (role-based)
CREATE POLICY ops_all_access ON jobs
    FOR ALL USING (
        EXISTS (SELECT 1 FROM auth.users WHERE auth.uid() = id AND raw_user_meta_data->>'role' = 'ops')
    );

-- ── Updated_at triggers ────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER customers_updated_at BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER jobs_updated_at BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER quotes_updated_at BEFORE UPDATE ON quotes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
