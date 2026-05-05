-- Add metadata JSONB column to jobs table for ingestion/compute metadata
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS metadata JSONB;
