# Integrate local AI fallback

Stage: MVP

Labels: ai, offline, mvp

## Description
Allow running AI generation without external API via local models (DeepSeek, Ollama) when configured.

## Goals

- Config path for selecting provider (OpenAI/local)
- Portable codegen interface that supports local LLMs

## Acceptance Criteria

- AI gen works in offline mode with minimal setup
- Docs include instructions for switching providers
