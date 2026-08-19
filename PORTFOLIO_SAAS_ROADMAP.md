# Product and SaaS Roadmap

## Product direction

Position the app as a **private document intelligence workspace** for students,
job seekers, researchers, and knowledge workers. The promise is simple: upload
your material, get plain-language answers, and inspect the source behind every
claim. This is more approachable than presenting it as an “agentic RAG demo”.

## Current increment

This repository remains a prototype deployment, but now has safer browser
origin defaults, a single-pass streaming graph (avoiding duplicate LLM work),
and a first-use experience designed around the consumer job-to-be-done.

## Delivery plan

| Phase | Outcome | Key implementation work | Acceptance signal |
| --- | --- | --- | --- |
| 0. Stabilize | Reliable demo and clean portfolio story | Keep the current FastAPI + Streamlit stack; pin dependencies; add smoke tests; remove secrets from committed local files; use explicit CORS origins | A fresh clone starts with Docker and passes tests without a real provider key |
| 1. Identity and data | Private, persistent accounts | PostgreSQL, SQLAlchemy 2, Alembic, Clerk/Auth.js or FastAPI JWT, `users`, `workspaces`, `documents`, `conversations`, `messages` | A signed-in user can return to their files and cannot access another user's records |
| 2. Durable ingestion | Safe uploads that survive restarts | Object storage (S3/R2), async worker queue (Celery/Arq + Redis), document status tracking, retry/dead-letter handling, malware scanning | Large uploads show queued/indexing/ready states and recover after worker restarts |
| 3. Tenant-safe retrieval | Scale without leaking sources | Namespace vectors by workspace, enforce workspace filtering server-side, document deletion cascades, audit logs, quotas | Retrieval queries never accept a client-controlled tenant filter |
| 4. SaaS UX | A polished public web app | Replace Streamlit with Next.js + TypeScript, Tailwind, shadcn/ui; auth-aware routes, document library, conversation list, responsive mobile layout | Lighthouse and manual mobile checks pass; empty/loading/error states exist for every primary flow |
| 5. Monetization and operations | A product that can be operated | Stripe Checkout + webhooks, free/pro entitlements, usage metering, Sentry, OpenTelemetry, backups, admin views | Limits are enforced at the API and Stripe events are idempotent |

## Target boundaries

```text
Next.js web app
        │ HTTPS / JWT
FastAPI API ── auth + workspace policy + usage limits
        ├── PostgreSQL (users, metadata, conversations, billing)
        ├── Redis (rate limits, jobs, short-lived cache)
        ├── Object storage (original uploads)
        ├── Vector database (workspace-scoped chunks)
        └── Worker service (parse, embed, reindex)
```

Keep LangGraph behind a `ResearchService` interface so that HTTP routers do not
know about graph nodes, vector providers, or persistence details. Use repository
interfaces for documents, conversations, and usage. This allows local Chroma to
remain the development adapter while a managed vector store becomes the
production adapter.

## Data model to introduce first

- `users`: identity provider subject, email, display name, plan.
- `workspaces`: owner, name, settings; start with one personal workspace per user.
- `memberships`: user/workspace/role; enables teams later without a rewrite.
- `documents`: workspace ID, storage key, checksum, status, page/chunk counts.
- `conversations` and `messages`: workspace-owned history, citations stored as JSON.
- `usage_events`: workspace ID, event type, unit count, idempotency key.

Every database query and vector retrieval must receive the server-derived
workspace ID. Do not trust a workspace ID sent from the browser.

## Portfolio release checklist

- Use a memorable product name, one-sentence value proposition, and a demo PDF.
- Capture a 45–60 second demo: upload → status → cited answer → inspect source.
- Publish a small architecture diagram and explain the trade-off between local
  Chroma for the demo and a managed multi-tenant store for production.
- Add screenshots for desktop and mobile, plus a deployed read-only demo mode.
- Add a privacy note explaining storage, deletion, supported file types, and AI
  provider use before inviting real users.

## Guardrails

Authentication, billing, persistent storage, and multi-tenancy must ship as one
cohesive vertical slice. Adding only a login screen or client-side document
filter is not a security boundary. Until then, label this deployment as a
single-user demo and do not upload sensitive documents.
