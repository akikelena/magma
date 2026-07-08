# MAGMA Agent Cards

Every architectural element is catalogued here by modality,
inputs/outputs (pre/postconditions), mechanism, and failure behavior.
This document is that catalogue for the thirteen agents implemented in
this repository, plus the Knowledge Repository Agent's role as the
shared blackboard all of them read and write.

Modality follows a four-value taxonomy:

- **ALGORITHMIC** -- deterministic, no model in the loop.
- **LLM** -- autonomous, model-driven end to end.
- **HYBRID** -- a deterministic stage composed with an LLM or ML stage.
- **HUMAN** -- human-in-the-loop.

| # | Agent | Module | Modality |
|---|-------|--------|----------|
| 1 | Documentation Processing Agent | Phase 1 | ALGORITHMIC |
| 2 | Knowledge Repository Agent | Cross-cutting (blackboard) | ALGORITHMIC |
| 3 | Change Detection Agent | Phase 1 | HYBRID |
| 4 | Version Coherence Agent | Phase 1 | HYBRID |
| 5 | Task Orchestration Agent | Phase 2 | ALGORITHMIC |
| 6 | Prompt Generation Agent | Phase 2 | LLM |
| 7 | Code Generation Agent | Phase 2 | LLM |
| 8 | Automated Validation Agent | Phase 3 | ALGORITHMIC |
| 9 | Human Expert Agent | Phase 3 | HUMAN |
| 10 | Adaptive Learning Agent | Phase 4 | HYBRID |
| 11 | Explainability Agent | Phase 4 | LLM |
| 12 | Usage Feedback Agent | Phase 4 | ALGORITHMIC |
| 13 | Synthetic Testing Agent | Phase 4 | ALGORITHMIC |

---

## 1. Documentation Processing Agent

`magma/phase1_documentation_detection/documentation_processing_agent.py`

**Modality:** ALGORITHMIC (deterministic service).

**Precondition (input):** raw documentation retrieved from HTTP endpoints,
version-controlled repositories, or APIs (`RawDocument`: content,
source_url, vendor, version).

**Postcondition (output):** structured `DocumentationFragment` objects,
each carrying a semantic label and a deterministic, stable
`fragment_id` (SHA256 of source_url + heading_path + ordinal), so the same
logical fragment resolves to the same graph node identity across document
versions.

**Mechanism:** a three-stage pipeline -- (1) structure detection and
parsing into a normalized tree, (2) NER / dependency parsing / topic
modeling via a pluggable `NLPBackend` (heuristic default, spaCy-backed
in production), (3) fragment labeling.

**Failure behavior:** malformed or unparseable sources are recorded as a
`SkippedSource` with a reason and excluded from downstream processing
rather than passed through -- a parsing failure cannot silently propagate
into classification. This is part of the architecture's first line of
defense.

---

## 2. Knowledge Repository Agent

`magma/knowledge_repository/repository.py`

**Modality:** ALGORITHMIC (deterministic data-management service).

**Precondition (input):** a `WriteRequest` (key, value, attribution) for
writes; a key, node id, or similarity query for reads. Every other agent
in the architecture both reads and writes this agent -- it is the
blackboard's storage layer, not a passive database.

**Postcondition (output):** for writes, a versioned commit id; for reads,
the current (or a prior, versioned) value, graph node/edge, or a ranked
list of vector `SimilarityMatch` results.

**Mechanism:** composes a `GraphStore` (three node types: documentation
fragment, template component, semantic concept; four edge types:
dependency, generation, supersession, conflict) with a `VectorStore`
(cosine similarity, MRL embedding truncation to 1024 dimensions,
similarity threshold 0.78). Adds timestamped/attributed versioning on top
of both. `docker-compose.yml` provisions the live Neo4j/Weaviate backends;
`magma/scripts/bootstrap_schema.py` idempotently creates the Neo4j
uniqueness constraint/index and the Weaviate collection each store
assumes exists, and `magma/scripts/seed_knowledge_repository.py` inserts
real vendor documentation fragments into them.

**Failure behavior:** on detection of a downstream error, callers can roll
back to a prior versioned state rather than the repository silently
carrying forward corrupted state.

---

## 3. Change Detection Agent

`magma/phase1_documentation_detection/change_detection_agent.py`

**Modality:** HYBRID -- deterministic Myers-diff preprocessing composed
with an LLM chain-of-thought classification stage. Determinism and
flexibility are separated by stage rather than dissolved within one unit.

**Precondition (input):** two documentation versions of the same fragment
(`FragmentVersionPair`: fragment_id, old_text, new_text).

**Postcondition (output):** a filtered and grouped `ChangeSet`, each
`DetectedChange` carrying one of six severity labels (EDITORIAL, PATCH,
MINOR, MAJOR, DEPRECATION, BREAKING), a rationale, and a confidence score.

**Mechanism:** `difflib`-based diff computation and editorial-only
filtering first (deterministic), then an LLM call classifies the
remaining semantic changes along two underlying attributes (intent_shift,
compatibility_preservation) via `change_classification_prompt`.

**Failure behavior:** classifications below `confidence_threshold` (0.70)
are marked `is_edge_case=True` and escalated to human review via
`EscalationPolicy` rather than forced into a category. Security-critical
keyword matches escalate regardless of confidence. This realizes
"escalate rather than commit" for the first line of defense.

---

## 4. Version Coherence Agent

`magma/phase1_documentation_detection/version_coherence_agent.py`

**Modality:** HYBRID -- deterministic graph traversal composed with an
ML conflict predictor (XGBoost, heuristic fallback when unavailable).

**Precondition (input):** a classified `ChangeSet` and the knowledge
graph (`CoherenceRequest`).

**Postcondition (output):** a `CoherenceOutcome` carrying the affected
template ids, a topologically sequenced modification order, and either a
clear-to-proceed signal or a set of raised conflict flags.

**Mechanism:** locates affected templates via GENERATION edges from
changed fragments, propagates cascading impact via
`reverse_transitive_closure` over DEPENDENCY edges (the second line of
defense), fixes modification order via `topological_sort`, and
cross-checks with rule-based conflict detection (dependency cycles,
invariant violations, simultaneous modification) plus a weighted
ML conflict score.

**Failure behavior:** any flagged conflict suspends automated processing
for the affected changes and escalates to the Human Expert Agent with
explanatory context (`EscalationReason.CONFLICTING_SPECIFICATION`).

---

## 5. Task Orchestration Agent

`magma/phase2_artifact_generation/task_orchestration_agent.py`

**Modality:** ALGORITHMIC (deterministic service). The supervisor of the
architecture: it never holds model state and never calls another agent
directly, only reads the blackboard and returns a plan.

**Precondition (input):** a validated change set, post Version Coherence
Agent (`OrchestrationRequest`).

**Postcondition (output):** an `ExecutionPlan` -- an execution mode
(lightweight, comprehensive, or parallel) and an ordered list of
`TemplateTask`s, each carrying its maintenance category, generation mode,
generation rules, output formats, and architectural constraints.

**Mechanism:** performs all four analyses assigned to this agent,
independently rather than reusing another agent's computation:
change-scope analysis (localized vs. distributed, by affected node
count), change-type analysis (maps the six-category classification to a
maintenance category via a decision table), affected-artifact analysis
(traverses GENERATION edges from changed fragments to find templates,
then reads generation_rules/output_formats/architectural_constraints from
each template's graph-node properties), and dependency analysis
(`reverse_transitive_closure` + `topological_sort` over DEPENDENCY
edges to fix modification order, matching Version Coherence Agent's
traversal pattern but computed independently for planning purposes).

**Failure behavior:** a stage that fails its quality gate
(`evaluate_quality_gate`) is returned for rework within the
three-iteration limit; beyond that, the case is escalated to the Human
Expert Agent.

---

## 6. Prompt Generation Agent

`magma/phase2_artifact_generation/prompt_generation_agent.py`

**Modality:** LLM (autonomous).

**Precondition (input):** a `TemplateTask` and the current template
content (`PromptGenerationRequest`).

**Postcondition (output):** a `GeneratedPrompt` -- an assembled,
token-budget-fitted prompt for the Code Generation Agent, plus a flag for
whether a supporting exemplar was found.

**Mechanism:** assembles the prompt from five sources: annotated template
context pulled from the template's graph-node properties in the Knowledge
Repository, an LLM-compressed one-sentence statement of intent, the
task's generation rules / output formats / architectural constraints,
integrated metadata (framework conventions, coding standards, security
policy), and exemplars retrieved by semantic similarity (query defaults
to the task's compressed instructions when the caller supplies none).
Sections are greedily included, most important first, under
`token_budget`.

**Failure behavior:** if no sufficiently similar exemplar is retrieved,
the agent proceeds with the constraint specification alone
(`exemplar_absent=True`) and records the absence for the Usage Feedback
Agent rather than blocking generation.

---

## 7. Code Generation Agent

`magma/phase2_artifact_generation/code_generation_agent.py`

**Modality:** LLM (autonomous).

**Precondition (input):** an assembled prompt, a generation mode, and
(on rework) prior validation diagnostics (`CodeGenerationRequest`).

**Postcondition (output):** a `GeneratedArtifact` -- content,
documentation comments, and test cases -- or a structured failure with the
parse error, on success/failure respectively.

**Mechanism:** two-stage synthesis -- a reasoning stage (chain-of-thought,
no code emitted yet) followed by a synthesis stage that combines
prompt-chaining (the reasoning output feeds forward) and expert prompting
(a domain-specific system-prompt hint, e.g. Xtend template inheritance or
SQL dialect differences) to produce parsed JSON output. Operates in
targeted-modification mode (<30% of template lines affected) or
comprehensive-synthesis mode, as selected by the Task Orchestration
Agent.

**Failure behavior:** a synthesis-stage JSON parse failure is returned as
an unsuccessful `AgentResult` with the parse error as rationale, logged
with duration; the caller (Phase 3's rework loop) is responsible for
resubmission with diagnostics appended.

---

## 8. Automated Validation Agent

`magma/phase3_validation/automated_validation_agent.py`

**Modality:** ALGORITHMIC (deterministic service).

**Precondition (input):** a generated artifact and its test cases, plus
the current rework attempt number (`AutomatedValidationRequest`).

**Postcondition (output):** a `ValidationOutcome` with per-layer
`LayerResult`s and, on failure, the first failing layer's diagnostics.

**Mechanism:** applies six validators in ascending-cost order with
early-exit on first failure -- syntax (balanced-delimiter check;
production: Xtend syntax checking), style (line-length/trailing-whitespace;
production: custom rulesets), functional (for Python-shaped artifacts,
real sandboxed subprocess execution of the generated tests against the
generated content, with basic statement-coverage measurement; production:
JUnit against Xtend output), integration (pluggable no-op stand-in for a
Docker-containerized VDBMS harness), security (regex-based scan for
eval/exec usage, hardcoded secrets, disabled TLS verification, SQL string
concatenation), and performance (pluggable no-op stand-in for JMH).

**Failure behavior:** a failure at any layer returns the artifact for
regeneration with diagnostics; up to three autonomous rework attempts are
permitted (delegated to Task Orchestration Agent's
`evaluate_quality_gate`) before escalating to the Human Expert Agent.

---

## 9. Human Expert Agent

`magma/phase3_validation/human_expert_agent.py`

**Modality:** HUMAN (human-in-the-loop).

**Precondition (input):** an escalated case with diagnostic context
(episode id, `EscalationReason`, arbitrary context dict).

**Postcondition (output):** an approval or rejection decision with an
annotated rationale, recorded on the `ReviewTicket`.

**Mechanism:** works as a queue rather than a synchronous call, since a
human cannot be blocked on inside an automated pipeline: `process`
enqueues a `ReviewTicket` (status PENDING) and returns immediately; a
human later calls `submit_decision`, which resolves the ticket. One
shared queue instance is used across Phase 1 and Phase 3 escalation
paths, so a reviewer sees every pending case in one place. Triggered
under four conditions: low documentation-interpretation confidence,
validation failures exceeding the retry limit, conflicting
specifications, and security-critical changes.

**Failure behavior:** not applicable -- this agent is the terminal
authority for escalated cases; there is no further escalation path beyond
it.

---

## 10. Adaptive Learning Agent

`magma/phase4_improvement/adaptive_learning_agent.py`

**Modality:** HYBRID -- supervised learning composed with feedback-driven
policy learning.

**Precondition (input):** an episode's routing features and outcome
(`AdaptiveLearningRequest`) for policy updates; `PromptStructureFeatures`
(change-type ordinal, affected-entity count, security-relatedness) for
prompt-structure prediction; validated successful episodes for
fine-tuning dataset collection.

**Postcondition (output):** an `AdaptiveLearningOutcome` (updated routing
policy state) from `process`; a `PromptStructureLabel`
(MINIMAL/STANDARD/RICH) from `predict_prompt_structure`; a dispatched
fine-tuning batch (via a pluggable `FineTuningBackend`, no-op by default)
from `dispatch_fine_tuning`.

**Mechanism:** workflow-routing policy is a contextual bandit (LinUCB),
chosen over full reinforcement learning because each decision is a single
step with an immediately observed, attributable reward (success minus
time penalty minus intervention penalty), and available episode counts
are far below what policy-gradient RL needs for stable convergence.
Prompt-structure prediction uses a `DecisionTreeClassifier` (lazy
sklearn import) once at least 8 labeled examples are collected, with a
heuristic fallback below that. Updates are validated offline on held-out
sets and released through canary rollouts monitored for regressions.

**Failure behavior:** a reward regression during canary rollout triggers
automatic rollback to the prior policy, logged with diagnostic
information for subsequent analysis.

---

## 11. Explainability Agent

`magma/phase4_improvement/explainability_agent.py`

**Modality:** LLM (autonomous).

**Precondition (input):** an episode's audit trail and its validation
outcomes (`ExplanationRequest`).

**Postcondition (output):** an `ExplanationRecord` -- a structured,
human-readable rationale (5-8 sentences: why the modification was
required, which artifacts it affected, confidence given the validation
outcomes, and the reasoning path from documentation change to deployed
generator).

**Mechanism:** realizes the fifth line of defense -- explanation and
audit -- by producing a traceable rationale record for each maintenance
episode from its accumulated `AuditRecord`s, enabling post-hoc
identification of latent errors that escaped the preceding four lines.

**Failure behavior:** an episode with fewer than the minimum required
audit records is logged as incompletely explainable rather than given a
fabricated rationale.

---

## 12. Usage Feedback Agent

`magma/phase4_improvement/usage_feedback_agent.py`

**Modality:** ALGORITHMIC (deterministic service).

**Precondition (input):** a list of `AuditRecord`s (the audit trail).

**Postcondition (output):** a `FeedbackReport`: slowest actions by
frequency and by real measured duration, most-retried templates,
recurring failure patterns, and a per-change maintenance-success
probability estimate.

**Mechanism:** aggregates performance metrics, failure events, and
resource utilization from structured logging. Latency is real: agents
instrumented with `timed()` (Change Detection, Prompt Generation,
Explainability) report actual elapsed milliseconds, summed per action
into `slowest_actions_by_duration_ms`. Maintenance-success probability
uses regularized logistic regression for interpretability and calibration
on small labeled samples, substituted for a gradient-boosted model only
once episode count exceeds 200.

**Failure behavior:** not applicable -- this agent is read-only with
respect to the pipeline; it cannot itself cause a maintenance episode to
fail.

---

## 13. Synthetic Testing Agent

`magma/phase4_improvement/synthetic_testing_agent.py`

**Modality:** ALGORITHMIC (deterministic service).

**Precondition (input):** a list of `AdversarialScenario`s, each a named,
runnable stress condition against a copy of the pipeline configuration.

**Postcondition (output):** a `RobustnessReport` -- per-scenario survival
outcomes and an aggregate survival rate.

**Mechanism:** subjects the decision logic and error-handling mechanisms
to adversarial scenarios (malformed inputs, conflicting specifications,
resource exhaustion, etc.), operating offline on a copy of the
configuration/pipeline components so it never mutates production state.

**Failure behavior:** not applicable -- this agent operates offline on a
copy; a scenario "failing" (the system mishandling it) is the signal the
agent exists to surface, not an agent malfunction.
