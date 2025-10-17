# Basic caching + metadata schema

Stage: Pre-MVP

Labels: core, cache, protocol

## Description
Create a JSON cache storing kernel/backend decisions, including device hints and performance metadata.

## Goals

- Cache get/set/delete/clear APIs
- Enrich records with device_hint, driver_info, source_hash, timestamp
- Invalidation: all, by device substring, by backend name

## Acceptance Criteria

- `uhop cache list/show/delete/clear/invalidate` work end-to-end
- Cache keys used by optimizer and updated after runs
