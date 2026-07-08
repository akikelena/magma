# MAGMA: Multi-Agent System for Automated Maintenance of Template-Based Code Generators

[![CI](https://github.com/akikelena/magma/actions/workflows/ci.yml/badge.svg)](https://github.com/akikelena/magma/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

*(CI badge URL is a placeholder -- update `<your-username>` once this is pushed to GitHub.)*

MAGMA automates the maintenance of template-based code generators in response to evolving
external (vendor) documentation. It is realized as a supervisor-orchestrated blackboard
architecture: a shared, versioned Knowledge Repository is the blackboard, the Task
Orchestration Agent is the deterministic supervisor, and only the stages that require
semantic judgment are autonomous LLM agents.

## Implementation status

| Module | Status |
| --- | --- |
| Phase 1 -- Documentation Change Detection | **Implemented**: Documentation Processing, Knowledge Repository, Change Detection, Version Coherence agents. |
| Phase 2 -- Software Artifact Generation | **Implemented**: Task Orchestration, Prompt Generation, Code Generation agents. |
| Phase 3 -- Code Validation | **Implemented**: six-layer Automated Validation Agent, Human Expert Agent review queue, bounded corrective rework cycle. |
| Phase 4 -- System Improvement | **Implemented**: Explainability, Synthetic Testing, Usage Feedback, Adaptive Learning (LinUCB) agents. |

All four modules are wired together by `magma.orchestration.MagmaWorkflow`, which runs Phase 1 -> Phase 2 ->
Phase 3 sequentially over one shared `KnowledgeRepositoryAgent` blackboard (agents never call each other
directly), stops and returns `PENDING_HUMAN_REVIEW` if Phase 1 or Phase 3 escalates, and triggers Phase 4 on
completion.

## Project structure

```
magma/
  core/
    interfaces/        # Agent, MessageBus, LLMProvider contracts
    models/             # documentation, change, graph, template, validation data models
    governance/         # audit logging, escalation policy
    utils/               # config loading, observability
  knowledge_repository/  # the blackboard: Neo4j graph store + Weaviate vector store
  llm_providers/         # Anthropic + mock LLMProvider implementations
  phase1_documentation_detection/
    documentation_processing_agent.py
    change_detection_agent.py
    version_coherence_agent.py
    ...
  orchestration/         # Phase1Pipeline/Phase2Pipeline/Phase3Pipeline + MagmaWorkflow: wires all four modules together
  crewai_agents/         # CrewAI-orchestrated wiring (production coordination layer)
  config/                # agents.yaml, llm_models.yaml
  scripts/
    run_workflow.py       # CLI entry point: runs MagmaWorkflow end to end from a shell
  tests/
    unit/
    integration/         # exercises the full Phase 1 pipeline against real vendor SDK docs
docs/
  architecture/
    agent_cards.md         # modality, pre/postconditions, mechanism, failure behavior for every agent
```

## Getting started

```bash
poetry install
cp .env.example .env       # fill in API keys / DB URIs
pytest magma/tests/unit magma/tests/integration
```

`KnowledgeRepositoryAgent`, `ChangeDetectionAgent`, and friends all accept dependency-injected
backends (`InMemoryGraphStore`, `InMemoryVectorStore`, `MockLLMProvider`), so the full test
suite runs without a live Neo4j/Weaviate instance or an LLM API key. Swap in
`Neo4jGraphStore` / `WeaviateVectorStore` / `AnthropicProvider` for production use.

## Running the full workflow

```python
from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.llm_providers import AnthropicProvider
from magma.orchestration import MagmaWorkflow
from magma.phase1_documentation_detection import RawDocument

kr = KnowledgeRepositoryAgent()
llm = AnthropicProvider(api_key="...")
workflow = MagmaWorkflow(knowledge_repository=kr, llm_provider=llm)

result = workflow.run(
    old_document=RawDocument(content=old_docs, source_url="...", vendor="Milvus", version="v2.3.x"),
    new_document=RawDocument(content=new_docs, source_url="...", vendor="Milvus", version="v2.4.x"),
    template_contents={"search_template": current_template_source},
)
print(result.status, result.phase3.all_passed if result.phase3 else None)
```

Each module can also be run standalone via `Phase1Pipeline` / `Phase2Pipeline` / `Phase3Pipeline` for testing
or partial runs.

## Command-line usage

The repo is directly runnable, not just importable:

```bash
# Zero-setup demo: bundled Milvus fixture pair, scripted mock LLM, no API key needed.
python -m magma.scripts.run_workflow --demo

# Real run against your own documents and templates.
python -m magma.scripts.run_workflow \
    --old-doc old.md --new-doc new.md \
    --source-url https://example.com/docs/api --vendor acme \
    --old-version v1 --new-version v2 \
    --heading "API Reference/search" \
    --template search_template=path/to/current_search_template.py \
    --anthropic-api-key sk-...   # or set ANTHROPIC_API_KEY
```

`poetry install` also registers this as the `magma` console script (`magma --demo`).

Per-agent defaults (the Change Detection Agent's confidence threshold, its LLM model alias, the Automated Validation Agent's rework-attempt budget) are externalized in `config/agents.yaml` and `config/llm_models.yaml`, loaded via `magma.core.utils.config.load_config()`. The CLI reads them for its `--model`/`--confidence-threshold`/`--max-attempts` defaults (see `--help`); flags passed explicitly still override the config file per-run.

Synthetic Testing Agent has its own entry point, since it deliberately runs
offline against throwaway agent instances rather than the live pipeline:

```bash
python -m magma.scripts.run_synthetic_tests
```

## Running against live Neo4j / Weaviate

The full test suite and `--demo` above never require live infrastructure -- every
agent defaults to `InMemoryGraphStore`/`InMemoryVectorStore`. To exercise the real
Neo4j/Weaviate backends:

```bash
cp .env.example .env               # fill in NEO4J_PASSWORD at minimum
docker compose up -d
export $(grep -v '^#' .env | xargs)
python -m magma.scripts.bootstrap_schema             # constraint/index + Weaviate collection
python -m magma.scripts.seed_knowledge_repository    # inserts the real Milvus/Weaviate fixtures
```

`seed_knowledge_repository` runs `DocumentationProcessingAgent` over the bundled vendor
fixtures, writes each fragment as a graph node plus a vector embedding, and links two demo
template nodes to them via `GENERATION` edges -- so the resulting graph is immediately
usable by `run_workflow` or by `TaskOrchestrationAgent`'s own graph traversal.

## CrewAI orchestration

CrewAI v1.5.0 is the production coordination layer. `magma/crewai_agents/`
wraps each phase pipeline's `run` method as a CrewAI tool so the same MAGMA agents run under
CrewAI's `Crew`/`Task`/`Process` scheduling instead of `MagmaWorkflow`'s direct procedural
calls -- the agents keep sole ownership of their logic either way; only the orchestration
layer differs. This is an alternate execution mode of `MagmaWorkflow` itself, not a separate
demo:

```python
result = workflow.run_via_crewai(old_document, new_document, template_contents)
# same WorkflowResult shape as workflow.run(...)
```

`build_phase1_crew` / `build_phase2_crew` / `build_phase3_crew` / `build_phase4_crew` build
each phase's crew individually (useful for partial runs); `build_magma_crew` composes all
four with task-to-task context chaining, including the human-review early exit that skips
Phase 2/3. Requires `pip install crewai` (`poetry install --with crewai`); every other code
path in this repo runs without it.

### Configuring Agent/Crew attributes

Each `crewai.Agent`'s scheduling behavior (iteration/time/rate limits, retry policy, caching,
delegation, code execution) and each `crewai.Crew`'s behavior (memory, planning, caching,
rate limiting) are exposed as `CrewAgentSettings`/`CrewSettings` (`magma/crewai_agents/settings.py`),
sourced by default from `config/crewai.yaml`:

```python
from magma.crewai_agents import CrewAgentSettings, build_phase3_crew

# override just Phase 3's crew, e.g. to raise the sandboxed validator's time budget
crew = build_phase3_crew(pipeline, phase2_result, agent_settings=CrewAgentSettings(max_execution_time=300))

# or per-phase, for the composed crew
from magma.crewai_agents import build_magma_crew
crew = build_magma_crew(
    workflow, old_document, new_document, template_contents,
    agent_settings={"phase3": CrewAgentSettings(max_execution_time=300)},
)
```

Any phase/setting left unspecified falls back to `config/crewai.yaml`; any key in that file
falls back to the dataclass defaults in `settings.py`.

## Documentation change watching

`config/vendor_sources.yaml` registers real documentation entry points for MAGMA's five
evaluated VDBMSs (Milvus, Weaviate, Pinecone, Qdrant, Chroma). `DocumentWatcher`
(`magma/phase1_documentation_detection/document_watcher.py`) fetches a vendor's live docs via
`HttpDocumentFetcher`, compares the content hash against the last snapshot stored in the
Knowledge Repository, and persists the new snapshot -- reusing the blackboard's own
`MessageBus` interface (`write`/`read`/`history`) rather than a separate store. A page that
needs JS rendering to show real content (Qdrant today) is flagged `is_placeholder` instead of
being silently diffed against an empty scrape.

```bash
# One pass over every registered vendor, human-readable summary.
python -m magma.scripts.run_doc_watch --once

# Continuous polling, checking again every hour, only these two vendors.
python -m magma.scripts.run_doc_watch --interval-seconds 3600 --vendors milvus,weaviate

# Route the scan through the CrewAI-orchestrated Documentation Change Detector agent instead
# of calling DocumentWatcher/Phase1Pipeline directly.
python -m magma.scripts.run_doc_watch --once --use-crewai
```

The Knowledge Repository defaults to in-memory, so "last known" state doesn't survive past a
single process -- fine for `--interval-seconds` polling, not for repeated `--once` runs from
cron. Pass `--neo4j-uri`/`--neo4j-password` (and optionally `--weaviate-url`) to persist
snapshots in the real backends across separate invocations instead.

In `--use-crewai` mode, the Documentation Change Detector agent (`build_phase1_crew`) gets
four tools and real autonomy over the scan -- listing supported vendors, checking one, running
classification on whatever changed, and classifying an already-fetched pair directly -- rather
than one fixed tool per call. This is a deliberate, narrow exception to keeping autonomous
decision points to a minimum: deciding "should I re-check this vendor" is new orchestration
surface, not a route through Prompt Generation, Code Generation, Explainability, or Change
Detection's own classification stage -- the four components where MAGMA's design accepts
LLM autonomy -- and a wrong call here just means a vendor gets checked again next cycle rather
than a value-affecting production decision.

### Full pipeline: regenerating and validating templates

Detection alone tells you documentation changed; `--templates-config` is what makes the
pipeline actually maintain the generators. Map a vendor to one or more templates:

```yaml
# templates.yaml
milvus:
  - template_id: search_template
    path: /templates/milvus_search.xtend
    heading: "Milvus Collection.search Reference/Parameters"
```

```bash
python -m magma.scripts.run_doc_watch --interval-seconds 3600 --templates-config templates.yaml
```

A vendor with no entry in this file keeps the classify-only behavior described above. A vendor
with an entry gets the full `MagmaWorkflow` instead of bare `Phase1Pipeline`: Phase 2
regenerates each mapped template from the new documentation, Phase 3 validates the regenerated
content, and the template's file on disk is overwritten only if that validation actually
passed -- a regeneration that fails validation leaves the existing file untouched rather than
replacing working code with something MAGMA isn't confident in. `--use-crewai` stays
classification-only regardless of `--templates-config`: regeneration is a value-affecting
production action, not the bounded "should I re-check this vendor" decision the crewai agent's
autonomy is scoped to.

## Human review queue

Escalated cases (low classification confidence, conflicting specs, an exhausted validation
rework budget, security-critical changes) land in the Human Expert Agent's queue as
`ReviewTicket`s. The queue is in-memory by default -- fine for a library caller or a
long-lived process, useless for a `--once` CLI invocation, since the ticket becomes
unreachable the moment that process exits. Pass `--audit-log`/`--review-queue-path` to
`run_workflow`/`run_doc_watch` to persist both to disk, then act on them from a separate
process:

```bash
python -m magma.scripts.run_workflow --demo --review-queue-path reviews.json --audit-log audit.jsonl

python -m magma.scripts.review_queue --queue-path reviews.json --list
python -m magma.scripts.review_queue --queue-path reviews.json --show <ticket-id>
python -m magma.scripts.review_queue --queue-path reviews.json \
    --approve <ticket-id> --rationale "renamed param, backward compatible" --decided-by alice
python -m magma.scripts.review_queue --queue-path reviews.json \
    --reject <ticket-id> --rationale "breaks callers on the old parameter name"
```

`JsonFileTicketStore` rewrites the whole file atomically (write-to-temp, then rename) on every
enqueue/decision -- ticket volume is escalations, not high-throughput events, so a full
snapshot per mutation is simpler than reconciling an append log. Pending-queue depth is also
exported as the `magma_escalations_total{reason=...}` Prometheus counter when `--metrics-port`
is set, so a growing backlog shows up next to the other workflow metrics rather than only being
visible by polling `--list`.

## Docker

`Dockerfile` packages the CLI entry points (`run_workflow`, `run_doc_watch`, `review_queue`,
`bootstrap_schema`, `seed_knowledge_repository`) as a runnable image -- a two-stage build so
build tooling never ships in the final image, installing the core dependencies plus the "web"
and "observability" poetry groups (`requests`, `prometheus-client`) explicitly, since plain
`pip install .` doesn't see poetry groups and both CLIs need them to do real work. The heavier
"crewai"/"ml" groups stay opt-in:

```bash
docker build -t magma .
docker build -t magma --build-arg EXTRAS="crewai spacy scikit-learn xgboost" .   # opt in
```

`docker-compose.yml` runs the full stack -- Neo4j, Weaviate, and a `magma` service running
`run_doc_watch --interval-seconds 3600` by default, with `AUDIT_LOG_PATH`/`REVIEW_QUEUE_PATH`
pointed at a persisted `/data` volume and a healthcheck that probes its own `/metrics` endpoint:

```bash
cp .env.example .env               # fill in NEO4J_PASSWORD, ANTHROPIC_API_KEY, ...
docker compose up -d
docker compose run --rm magma magma.scripts.bootstrap_schema
docker compose run --rm magma magma.scripts.seed_knowledge_repository

# One-off commands reuse the same image/network without touching the long-running watcher:
docker compose run --rm magma magma.scripts.run_workflow --demo
docker compose run --rm magma magma.scripts.review_queue --queue-path /data/reviews.json --list
```

To have the watcher regenerate and validate real templates (not just classify), mount a
templates directory and set `TEMPLATES_CONFIG_PATH` -- both are commented out in
`docker-compose.yml` by default so a plain `docker compose up` doesn't require one to exist.

`run_doc_watch` treats `docker stop`'s SIGTERM the same as Ctrl-C, finishing the in-flight
cycle's output and exiting cleanly instead of being hard-killed once Docker's stop grace period
expires.

`.github/workflows/ci.yml` runs `flake8` and the full unit+integration suite against Python
3.10/3.11 on every push/PR, then builds the Docker image and validates `docker-compose.yml`.

## Architecture documentation

See `docs/architecture/` for architectural decision records and the agent-card catalog.

## Contributing

See `CONTRIBUTING.md` for dev setup, running tests/lint, and PR expectations. Please report
security vulnerabilities per `SECURITY.md` rather than as a public issue.

## License

MIT -- see `LICENSE`.
