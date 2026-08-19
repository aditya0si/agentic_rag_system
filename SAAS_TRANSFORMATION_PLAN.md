# Agentic RAG → SaaS Transformation Plan
## From Research Prototype to Production-Grade Consumer SaaS

---

## 1. Executive Summary

This plan transforms the current Agentic RAG Research Assistant from a functional research prototype into a **production-grade SaaS product** that is:
- **Technologically useful** for the average consumer (students, researchers, professionals)
- **Aesthetically impressive** enough for portfolio showcase
- **Architecturally sound** for real-world deployment and scaling

**Current State:** FastAPI backend + Streamlit frontend + local ChromaDB + in-memory sessions  
**Target State:** Multi-tenant SaaS with modern React frontend, PostgreSQL + cloud vector DB, OAuth/auth, Stripe billing, and cloud-native deployment.

---

## 2. Current State Analysis

### What's Working Well ✅
| Area | Assessment |
|------|-----------|
| Agent Pipeline | Solid 5-agent LangGraph pipeline (Query Rewriter → Retriever → Relevance Grader → Answer Generator → Hallucination Checker) |
| Backend API | Clean FastAPI with Pydantic schemas, OpenAPI docs, structured logging |
| Security | Input validation, prompt injection detection, sanitization |
| Testing | pytest suite with unit/integration tests, RAGAS evaluation |
| CI/CD | GitHub Actions with lint, test, build, deploy stages |
| Docker | Multi-stage builds, health checks, compose orchestration |
| Code Quality | ruff, mypy, structlog logging, tenacity retries |

### Critical Gaps for SaaS ❌
| Gap | Impact | Priority |
|-----|--------|----------|
| **No Authentication** | Anyone can use it; no user accounts, no data privacy | P0 |
| **In-Memory Sessions** | All chat history & uploads lost on restart | P0 |
| **Streamlit Frontend** | Not suitable for SaaS; limited customization, poor mobile UX | P0 |
| **No Persistent Database** | No user profiles, conversation history, document metadata | P0 |
| **Local ChromaDB Only** | Single-node, file-based; won't scale across instances | P1 |
| **No Multi-tenancy** | Documents isolated by session only, not by user/org | P0 |
| **No Billing/Subscriptions** | Can't monetize; no free tier vs paid tier logic | P1 |
| **No Real-time Streaming** | Chat feels slow; UX inferior to ChatGPT/Claude | P1 |
| **No API for Developers** | Missing secondary revenue stream | P2 |
| **No Admin Dashboard** | Can't manage users, monitor usage, view metrics | P2 |

---

## 3. Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Next.js App │  │  React Admin │  │  Mobile PWA  │  │  Developer API   │ │
│  │  (Consumer)  │  │  Dashboard   │  │  (Future)    │  │  (API Keys)      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
└─────────┼─────────────────┼─────────────────┼───────────────────┼───────────┘
          │                 │                 │                   │
          └─────────────────┴─────────────────┴───────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Cloudflare /      │
                         │   AWS CloudFront    │
                         │   (CDN + WAF)       │
                         └──────────┬──────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌──────────────────────────────┴────────────────────────────────────────┐  │
│  │                        FastAPI Backend (Kubernetes/ECS)                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │  │
│  │  │  Auth    │ │ Document │ │  Chat    │ │  Admin   │ │  Billing     │  │  │
│  │  │  Router  │ │  Router  │ │  Router  │ │  Router  │ │  Webhooks    │  │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │  │
│  │       └─────────────┴────────────┴────────────┴────────────────┘        │  │
│  │                         │                                              │  │
│  │              ┌──────────▼──────────┐                                   │  │
│  │              │   LangGraph Agent   │                                   │  │
│  │              │      Pipeline       │                                   │  │
│  │              └──────────┬──────────┘                                   │  │
│  └─────────────────────────┼─────────────────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────────┐
│                       DATA LAYER                                             │
│  ┌─────────────────────────┼─────────────────────────────────────────────┐  │
│  │  ┌──────────────┐  ┌────┴───────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  │  PostgreSQL  │  │   Redis    │  │  Pinecone /  │  │   S3/R2     │  │  │
│  │  │  (Users,     │  │  (Sessions,│  │  Weaviate    │  │  (File      │  │  │
│  │  │   Billing,   │  │   Cache,   │  │  (Vector     │  │   Storage)  │  │  │
│  │  │   Convos)    │  │   Queues)  │  │   Search)    │  │             │  │  │
│  │  └──────────────┘  └────────────┘  └──────────────┘  └─────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Phases

---

### Phase 1: Backend Foundation — Auth, DB & Multi-Tenancy (Weeks 1-3)

**Goal:** Make the backend SaaS-ready with proper user management and persistent storage.

#### 1.1 Database Layer
- **Migrate from in-memory to PostgreSQL**
  - Users table (id, email, name, avatar, created_at, tier)
  - Conversations table (id, user_id, title, created_at, updated_at)
  - Messages table (id, conversation_id, role, content, citations, agent_trace, created_at)
  - Documents table (id, user_id, filename, doc_type, page_count, chunk_count, storage_path, created_at)
  - Document_Chunks reference table (for metadata joins)
  - Usage_Logs table (for analytics & billing metering)

- **Add SQLAlchemy 2.0 + Alembic migrations**
  - Async session management
  - Repository pattern for clean data access
  - Connection pooling with pgBouncer for production

#### 1.2 Authentication & Authorization
- **OAuth 2.0 + JWT:**
  - Google OAuth (primary — easy signup)
  - GitHub OAuth (for developers)
  - Email/password with secure hashing (bcrypt/Argon2)
  - Magic link login (passwordless option)

- **JWT Access + Refresh Token Pattern:**
  - Short-lived access tokens (15 min)
  - Long-lived refresh tokens (7 days) in httpOnly cookies
  - Token revocation endpoint

- **RBAC (Role-Based Access Control):**
  - `free` — limited uploads, limited queries/day
  - `pro` — higher limits, priority processing
  - `admin` — full dashboard access

#### 1.3 Multi-Tenancy
- **Row-level security via `user_id` columns**
- All queries scoped to authenticated user
- Document isolation: users can only see their own uploads
- Conversation isolation: private by default

#### 1.4 Session Management
- **Redis for session state + caching**
  - Replace `_sessions_store` dict with Redis
  - Session TTL (auto-cleanup)
  - Rate limiting per user (not just per IP)
  - Chat history caching for fast retrieval

#### 1.5 Configuration Management
- **Pydantic Settings v2** with validation
- Environment-specific configs (dev/staging/prod)
- Secrets management (AWS Secrets Manager / HashiCorp Vault)

**Deliverables:**
- [ ] PostgreSQL schema with migrations
- [ ] OAuth + JWT auth system
- [ ] User registration/login flow
- [ ] All routes protected with auth middleware
- [ ] Redis session cache
- [ ] Updated tests with auth fixtures

---

### Phase 2: RAG Pipeline Modernization (Weeks 3-5)

**Goal:** Make the AI pipeline more powerful, faster, and cost-efficient.

#### 2.1 Vector Store Upgrade
- **Replace local ChromaDB with cloud vector DB:**
  - **Option A: Pinecone** — Managed, fast, metadata filtering, hybrid search
  - **Option B: Weaviate** — Open-source, GraphQL interface, modular
  - **Option C: Qdrant** — Rust-based, fast, good self-hosted option
  - **Recommendation:** Start with **Pinecone** for zero-ops, migrate to Qdrant later if costs matter

- **Multi-tenant vector collections:**
  - Namespace per user OR metadata filtering by user_id
  - Efficient deletion when user deletes documents

#### 2.2 Embedding Upgrade
- **Upgrade from `all-MiniLM-L6-v2` to better models:**
  - `text-embedding-3-small` (OpenAI) — best quality/cost
  - `BAAI/bge-large-en-v1.5` (local, high quality)
  - Make it configurable per user tier

#### 2.3 Streaming Responses
- **SSE (Server-Sent Events) for real-time chat feel:**
  - Stream tokens as they're generated
  - Show live agent trace ("Thinking..." → "Searching..." → "Analyzing...")
  - Typing indicator effect
  - Current `/stream` router is basic — expand it to full streaming RAG

#### 2.4 Pipeline Improvements
- **Parallel agent execution where possible:**
  - Query Rewriter + Intent Classifier in parallel
  - Batch relevance grading (currently sequential with 2s sleep!)
  - Remove `time.sleep(2)` in grade_node — use batch LLM calls

- **Add missing agents:**
  - **Intent Classifier** — Route to different pipelines (Q&A, Summarize, Compare)
  - **Query Expander** — Generate sub-questions for complex queries (already have file, integrate)
  - **Citations Extractor** — More precise citation mapping (link specific sentences to chunks)
  - **Confidence Scorer** — Add confidence scores to answers

- **Caching Layer:**
  - Redis cache for frequent queries (exact match + semantic cache)
  - Embed cache to avoid re-embedding same queries
  - LLM response cache for identical contexts

#### 2.5 Cost Optimization
- **Smart LLM Routing:**
  - Simple queries → cheaper model (GPT-4o-mini / Gemini Flash)
  - Complex queries → stronger model (GPT-4o / Gemini Pro)
  - Grading tasks → cheapest capable model

- **Token Usage Tracking:**
  - Per-request token counting
  - User-level usage dashboards
  - Billing integration ready

**Deliverables:**
- [ ] Cloud vector DB integration
- [ ] Full SSE streaming pipeline
- [ ] Batch relevance grading (remove sleep delays)
- [ ] Query cache with Redis
- [ ] Token usage tracking

---

### Phase 3: Frontend Overhaul — Modern React Application (Weeks 5-8)

**Goal:** Replace Streamlit with a stunning, interactive, portfolio-worthy frontend.

#### 3.1 Tech Stack Selection
```
Framework:     Next.js 15 (App Router) + React 19
Language:      TypeScript 5.5
Styling:       Tailwind CSS 3.4 + shadcn/ui
Animation:     Framer Motion
State:         Zustand (client) + React Query/TanStack Query (server)
Auth:          NextAuth.js v5 (Auth.js)
Chat UI:       Custom built (not a library — portfolio impression matters)
Charts:        Tremor / Recharts (for usage dashboards)
Icons:         Lucide React
Fonts:         Inter (body) + JetBrains Mono (code)
```

#### 3.2 Design System
- **Color Palette:**
  - Primary: Indigo 600 (`#4F46E5`) — trustworthy, professional
  - Secondary: Emerald 500 (`#10B981`) — success, AI-powered
  - Accent: Amber 500 (`#F59E0B`) — warnings, highlights
  - Surface: Slate 50 → Slate 900 gradient system
  - Dark mode support (critical for portfolio appeal)

- **Typography:**
  - Inter for all UI text (clean, modern)
  - JetBrains Mono for code/chunks/technical content
  - Responsive type scale

- **Spacing & Radius:**
  - Consistent 4px grid system
  - Large radius (12-16px) for cards, buttons
  - Generous whitespace (premium feel)

#### 3.3 Page Structure
```
/                     → Landing page (marketing)
/app                  → Main application (auth required)
/app/chat             → Chat interface (core experience)
/app/library          → Document management
/app/history          → Conversation history
/app/settings         → User settings, API keys, billing
/admin                → Admin dashboard (admin only)
/pricing              → Pricing page
/docs                 → API documentation
/auth/login           → Login/Signup
/auth/callback        → OAuth callback
```

#### 3.4 Landing Page (Portfolio Star)
- **Hero Section:**
  - Animated gradient background
  - Large headline: "Your Documents, Understood"
  - Subheadline with typewriter effect
  - CTA buttons: "Start Free" / "View Demo"
  - Live demo widget (try without signup)

- **Feature Grid:**
  - Animated cards showing the 5-agent pipeline
  - Hover effects with micro-interactions
  - Statistics counters ("92% Faithfulness Score")

- **Interactive Demo:**
  - Embedded mini-chat that works with pre-loaded sample documents
  - No signup required — instant gratification

- **Social Proof:**
  - Testimonial carousel
  - "Trusted by researchers at..." (your university/company)
  - GitHub stars counter (live)

- **Pricing Section:**
  - Three-tier cards (Free / Pro / Enterprise)
  - Toggle monthly/annual
  - Feature comparison table
  - Stripe checkout integration

#### 3.5 Chat Interface (Core Experience)
- **Layout:**
  - Left sidebar: conversation list, document library
  - Center: chat thread
  - Right panel (collapsible): citation inspector, agent trace

- **Message Bubbles:**
  - User: clean, right-aligned, subtle gradient
  - Assistant: left-aligned, with animated avatar
  - Streaming text with typewriter effect
  - Syntax highlighting for code blocks

- **Agent Trace Visualization:**
  - Real-time step indicators (not just text)
  - Animated progress: Query → Search → Grade → Generate → Verify
  - Each step expands to show details (latency, tokens, chunks count)
  - Graph visualization of the pipeline (D3.js or React Flow)

- **Citations:**
  - Inline clickable citations `[1]`, `[2]`
  - Hover preview of source chunk
  - Click to open side panel with full context
  - Highlight exact text in source

- **Document Upload:**
  - Drag & drop zone with animated feedback
  - Upload progress with chunking visualization
  - Document cards with page count, processing status
  - Thumbnail generation for PDFs

- **Mobile Responsive:**
  - Bottom sheet for sidebar
  - Swipe gestures
  - Collapsible panels

#### 3.6 Dashboard Pages
- **Conversation History:**
  - Searchable list with preview
  - Pin/star important conversations
  - Bulk delete, export (JSON, MD, PDF)

- **Document Library:**
  - Grid/list view toggle
  - Search within documents
  - Document preview (PDF viewer)
  - Metadata: upload date, chunk count, last queried

- **Settings:**
  - Profile management
  - Theme toggle (light/dark/system)
  - API key generation (for developer tier)
  - Subscription management (Stripe Customer Portal)
  - Usage analytics (charts: queries/day, tokens used, documents uploaded)

**Deliverables:**
- [ ] Next.js project scaffold with design system
- [ ] Landing page with animations
- [ ] Full chat interface with streaming
- [ ] Document upload & library
- [ ] Auth flows (login/signup)
- [ ] Settings & billing pages
- [ ] Dark mode support
- [ ] Mobile responsive design

---

### Phase 4: SaaS Business Layer (Weeks 8-10)

**Goal:** Add everything needed to actually run this as a business.

#### 4.1 Subscription Management (Stripe)
- **Pricing Tiers:**
  | Tier | Price | Uploads | Queries/Day | Models | Features |
  |------|-------|---------|-------------|--------|----------|
  | Free | $0 | 3 docs, 50 pages each | 10 | Gemini Flash | Basic RAG |
  | Pro | $12/mo | 50 docs, 500 pages | 100 | GPT-4o / Gemini Pro | Advanced agents, streaming |
  | Team | $49/mo | Unlimited | 500 | All models | Collaboration, admin, API access |

- **Stripe Integration:**
  - Checkout sessions
  - Customer Portal (self-service)
  - Webhook handlers for subscription events
  - Usage-based billing (overages)
  - Trial periods (14 days Pro)

#### 4.2 Usage Quotas & Enforcement
- Middleware that checks user tier before processing
- Soft limits (warn) + hard limits (block)
- Real-time usage counters in Redis
- Daily/weekly reset logic

#### 4.3 API for Developers
- **API Key Management:**
  - Generate/revoke keys in settings
  - Key prefix + hashed storage
  - Rate limits per key

- **RESTful Endpoints:**
  ```
  POST /api/v1/documents       → Upload
  GET  /api/v1/documents       → List
  DELETE /api/v1/documents/:id → Delete
  POST /api/v1/chat            → Ask (streaming SSE)
  GET  /api/v1/conversations   → List history
  GET  /api/v1/usage           → Usage stats
  ```

- **SDK (Optional but impressive):**
  - Python SDK: `pip install agentic-rag-sdk`
  - TypeScript SDK: `npm install @agentic-rag/sdk`

#### 4.4 Admin Dashboard
- **User Management:**
  - List all users with filters
  - View subscription status
  - Impersonate user (for support)
  - Ban/suspend accounts

- **System Analytics:**
  - Daily active users (DAU)
  - Total queries, avg latency
  - Model usage breakdown
  - Revenue metrics
  - Error rates & alerts

- **Content Moderation:**
  - Flagged conversations review
  - Document abuse detection
  - Prompt injection logs

**Deliverables:**
- [ ] Stripe integration with all tiers
- [ ] Usage quota middleware
- [ ] Developer API with keys
- [ ] Admin dashboard

---

### Phase 5: DevOps & Cloud Deployment (Weeks 10-11)

**Goal:** Production-ready, scalable, cost-effective infrastructure.

#### 5.1 Infrastructure as Code
- **Terraform / Pulumi** for:
  - AWS VPC, subnets, security groups
  - ECS Fargate (backend containers)
  - RDS PostgreSQL
  - ElastiCache Redis
  - S3 for file storage
  - CloudFront CDN
  - Route53 DNS

#### 5.2 Container Orchestration
- **AWS ECS Fargate** or **Railway** or **Fly.io**
  - Auto-scaling based on CPU/memory + request count
  - Blue/green deployments
  - Health checks + auto-restart

#### 5.3 CI/CD Pipeline (Enhanced)
- **GitHub Actions:**
  - Lint → Test → Build → Deploy to staging → E2E tests → Deploy to prod
  - Frontend: Vercel preview deployments for PRs
  - Backend: Staging environment on every push to `develop`

#### 5.4 Monitoring & Observability
- **Logging:** Datadog or Grafana Loki (structured JSON logs)
- **Metrics:** Prometheus + Grafana dashboards
  - Request rate, latency, error rate (RED metrics)
  - LLM token usage, cost per request
  - Vector DB query latency
- **Alerting:** PagerDuty / Slack for:
  - Error rate > 1%
  - p95 latency > 10s
  - LLM API failures
  - DB connection pool exhaustion

#### 5.5 Security Hardening
- **HTTPS everywhere** (Let's Encrypt / ACM)
- **WAF rules** (CloudFlare / AWS WAF)
- **CSP headers** for frontend
- **Rate limiting** per user (not just IP)
- **Data encryption:**
  - At rest: RDS encryption, S3 SSE
  - In transit: TLS 1.3
- **Regular dependency scanning** (Snyk, Dependabot)
- **SOC 2 readiness** (audit logs, access controls)

**Deliverables:**
- [ ] Terraform IaC
- [ ] Staging + Production environments
- [ ] Auto-scaling configured
- [ ] Grafana dashboards
- [ ] Alerting rules
- [ ] Security audit passed

---

### Phase 6: Polish & Portfolio Optimization (Weeks 11-12)

**Goal:** Make this undeniably impressive for portfolio and user acquisition.

#### 6.1 Performance Optimization
- **Frontend:**
  - Lighthouse score 95+ (performance, accessibility, SEO, best practices)
  - Code splitting, lazy loading
  - Image optimization (Next.js Image)
  - Service Worker for offline chat history

- **Backend:**
  - Response caching (CDN for static, Redis for dynamic)
  - Connection pooling optimization
  - Query plan analysis for PostgreSQL
  - Vector DB index tuning

#### 6.2 SEO & Marketing
- **Landing Page SEO:**
  - Meta tags, Open Graph, Twitter Cards
  - Structured data (JSON-LD)
  - Blog with AI/RAG content (drives organic traffic)
  - Sitemap + robots.txt

- **Social Features:**
  - Share conversation (public link with read-only view)
  - Export as beautiful PDF report
  - "Powered by Agentic RAG" badge

#### 6.3 Onboarding Flow
- **Interactive Tutorial:**
  - First-visit guided tour (React Joyride)
  - Pre-loaded sample documents (demo mode)
  - Tooltips explaining each agent

- **Empty States:**
  - Beautiful illustrations (not boring text)
  - Clear CTAs at every step

#### 6.4 Accessibility (a11y)
- WCAG 2.1 AA compliance
- Keyboard navigation
- Screen reader support
- Focus management
- Color contrast ratios

#### 6.5 Documentation
- **README overhaul:**
  - Architecture diagram (updated)
  - Demo GIFs/videos
  - Local dev setup (one-command with Docker)
  - Contributing guide

- **API Documentation:**
  - OpenAPI/Swagger UI
  - Postman collection
  - Code examples in multiple languages

- **Architecture Decision Records (ADRs):**
  - Why Next.js over Vue?
  - Why Pinecone over Weaviate?
  - Document technical decisions

**Deliverables:**
- [ ] Lighthouse 95+ score
- [ ] SEO-optimized landing page
- [ ] Interactive onboarding
- [ ] Full documentation
- [ ] Portfolio-ready demo video

---

## 5. Tech Stack Summary

| Layer | Current | Target | Rationale |
|-------|---------|--------|-----------|
| **Frontend** | Streamlit | Next.js 15 + React 19 | SSR, SEO, portfolio quality |
| **Styling** | Inline CSS | Tailwind + shadcn/ui | Design system consistency |
| **Backend** | FastAPI | FastAPI (keep) | Excellent, keep it |
| **Database** | None | PostgreSQL 16 + SQLAlchemy 2 | ACID, relational data |
| **Migrations** | None | Alembic | Schema versioning |
| **Cache** | None | Redis 7 | Sessions, rate limits, query cache |
| **Vector DB** | ChromaDB local | Pinecone / Qdrant | Scalable, multi-tenant |
| **Auth** | None | NextAuth + JWT | OAuth, session management |
| **Payments** | None | Stripe | Subscriptions, billing |
| **Storage** | Local temp | AWS S3 / Cloudflare R2 | Scalable file storage |
| **Deployment** | Docker Compose | ECS Fargate + Vercel | Auto-scaling, managed |
| **IaC** | None | Terraform | Reproducible infrastructure |
| **Monitoring** | Basic logs | Grafana + Prometheus + Loki | Production observability |
| **CI/CD** | GitHub Actions | GitHub Actions (enhanced) | Preview deploys, E2E tests |

---

## 6. File Structure (Target)

```
agentic-rag-saas/
├── 📁 apps/
│   ├── 📁 web/                          # Next.js frontend
│   │   ├── app/                         # App Router
│   │   │   ├── (marketing)/             # Public pages
│   │   │   │   ├── page.tsx             # Landing page
│   │   │   │   ├── pricing/page.tsx
│   │   │   │   └── docs/page.tsx
│   │   │   ├── (app)/                   # Authenticated app
│   │   │   │   ├── chat/page.tsx
│   │   │   │   ├── library/page.tsx
│   │   │   │   ├── history/page.tsx
│   │   │   │   └── settings/page.tsx
│   │   │   ├── api/auth/[...nextauth]/  # Auth.js API route
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── ui/                      # shadcn/ui components
│   │   │   ├── chat/
│   │   │   ├── layout/
│   │   │   └── marketing/
│   │   ├── lib/
│   │   │   ├── api.ts                   # API client
│   │   │   ├── auth.ts                  # Auth configuration
│   │   │   └── utils.ts
│   │   ├── hooks/
│   │   ├── stores/                      # Zustand stores
│   │   └── types/
│   │
│   └── 📁 admin/                        # Admin dashboard (optional separate app)
│
├── 📁 packages/
│   ├── 📁 ui/                           # Shared UI component library
│   ├── 📁 config/                       # Shared configs (eslint, tsconfig)
│   └── 📁 types/                        # Shared TypeScript types
│
├── 📁 backend/                          # FastAPI (refactored)
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dependencies.py              # FastAPI dependencies (DB, auth)
│   │   ├── database.py                  # SQLAlchemy setup
│   │   ├── models/                      # SQLAlchemy models
│   │   │   ├── user.py
│   │   │   ├── conversation.py
│   │   │   ├── document.py
│   │   │   └── subscription.py
│   │   ├── schemas/                     # Pydantic schemas
│   │   ├── routers/
│   │   │   ├── auth.py
│   │   │   ├── chat.py
│   │   │   ├── documents.py
│   │   │   ├── billing.py
│   │   │   └── admin.py
│   │   ├── services/
│   │   │   ├── auth_service.py
│   │   │   ├── chat_service.py
│   │   │   ├── document_service.py
│   │   │   └── billing_service.py
│   │   ├── core/
│   │   │   ├── agents/                  # LangGraph pipeline
│   │   │   ├── vector_store.py
│   │   │   ├── ingestion.py
│   │   │   └── cache.py
│   │   ├── middleware/
│   │   └── tasks/                       # Background tasks (Celery/ARQ)
│   ├── alembic/
│   ├── tests/
│   └── Dockerfile
│
├── 📁 infra/                            # Terraform / Pulumi
│   ├── modules/
│   ├── staging/
│   └── production/
│
├── 📁 docs/                             # Documentation
│   ├── architecture/
│   ├── api/
│   └── adrs/
│
├── docker-compose.yml                   # Local dev stack
├── turbo.json                           # Turborepo config
├── pnpm-workspace.yaml
└── README.md
```

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| LLM costs spiral | Token tracking, smart model routing, usage quotas, caching |
| Vector DB costs | Start with Pinecone free tier; monitor; consider Qdrant self-hosted |
| Auth complexity | Use Auth.js/NextAuth — battle-tested, reduces custom code |
| Scope creep | Ship Phase 1-3 first; 4-6 can come in v2 |
| Stripe integration complexity | Use Stripe Checkout + Customer Portal (minimal custom UI) |
| Performance at scale | Load test with k6; implement caching aggressively |
| Data privacy concerns | Clear privacy policy; SOC 2 roadmap; data deletion guarantees |

---

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Lighthouse Score | 95+ all categories |
| Time to First Byte | < 200ms |
| Chat Response Time | < 3s end-to-end (non-streaming) |
| API Uptime | 99.9% |
| Test Coverage | 80%+ |
| Free → Paid Conversion | 5%+ |
| NPS Score | 50+ |

---

## 9. Quick Start (Phase 1-3 Only)

If you want to move fast and get something portfolio-worthy in 6-8 weeks, focus ONLY on:
1. **Auth + DB** (PostgreSQL + NextAuth)
2. **Vector DB** (Pinecone free tier)
3. **Modern Frontend** (Next.js + beautiful chat UI)
4. **Streaming** (SSE for real-time feel)

Skip Stripe, skip admin dashboard, skip Terraform — use Railway or Render for deployment. You can always add business features later.

---

*Plan created for transforming Agentic RAG Research Assistant into a consumer SaaS product and portfolio showcase.*
