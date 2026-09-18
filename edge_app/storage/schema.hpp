#pragma once
namespace edge_app::detail
{
inline constexpr int schema_version = 2;
inline constexpr char create_schema[] = R"sql(
CREATE TABLE attendance_outbox (
    event_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    occurred_ms INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_ms INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX retry_due ON attendance_outbox(next_ms, occurred_ms);
CREATE TABLE dead_letter (
    event_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE gallery_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    version TEXT NOT NULL,
    etag TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_hash TEXT NOT NULL
);
CREATE TABLE gallery_template (
    template_id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL,
    embedding BLOB NOT NULL CHECK(length(embedding)=2048)
);
PRAGMA user_version=1;
)sql";
// Preserve pending retries when upgrading an existing v1 database.
inline constexpr char migrate_v1_to_v2[] = R"sql(
CREATE TABLE delivery_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    retry_not_before_ms INTEGER NOT NULL
);
INSERT INTO delivery_state
SELECT 1, COALESCE(MAX(next_ms),0) FROM attendance_outbox;
PRAGMA user_version=2;
)sql";
} // namespace edge_app::detail
