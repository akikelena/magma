# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/) once tagged releases start.

## [Unreleased]

### Added
- OSS project scaffolding: `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, GitHub issue/PR templates, `dependabot.yml`, `poetry.lock`.

### Fixed
- `numpy`/`spacy` and Python-version/`crewai` constraints in `pyproject.toml` conflicted enough
  to make `poetry lock` fail outright for the optional `ml`/`crewai` groups.

## [0.1.0] - 2026-07-08

Initial implementation of the full MAGMA architecture: a supervisor-orchestrated blackboard
system that detects vendor documentation changes and maintains template-based code generators
in response.

### Added
- **Phase 1 -- Documentation Change Detection**: Documentation Processing, Knowledge Repository
  (Neo4j graph + Weaviate vector, in-memory fallbacks), Change Detection (deterministic diff +
  LLM classification), Version Coherence agents.
- **Phase 2 -- Software Artifact Generation**: Task Orchestration, Prompt Generation, Code
  Generation agents.
- **Phase 3 -- Code Validation**: six-layer Automated Validation Agent, Human Expert Agent
  review queue (with `JsonFileTicketStore` persistence and a `review_queue` CLI), bounded
  corrective rework cycle.
- **Phase 4 -- System Improvement**: Explainability, Synthetic Testing, Usage Feedback, and
  Adaptive Learning (LinUCB) agents.
- `magma.orchestration.MagmaWorkflow` wiring all four modules end to end, plus a
  `run_via_crewai` alternate execution mode under CrewAI's `Crew`/`Task`/`Process` scheduling.
- Real vendor documentation acquisition (`DocumentWatcher`, `HttpDocumentFetcher`) for Milvus,
  Weaviate, Pinecone, Qdrant, and Chroma, with a scheduled/on-demand `run_doc_watch` CLI that
  optionally regenerates and validates mapped templates end to end (`--templates-config`).
- Governance: append-only `AuditLogger`, `EscalationPolicy` (four escalation triggers),
  Prometheus metrics (episode/validation/phase-duration/escalation counters).
- `run_workflow`/`run_doc_watch`/`review_queue`/`bootstrap_schema`/`seed_knowledge_repository`
  CLI entry points; `Dockerfile` + `docker-compose.yml` for a full containerized deployment.
- CI (`.github/workflows/ci.yml`): lint + full unit/integration suite on Python 3.10/3.11,
  Docker image build and compose validation.
