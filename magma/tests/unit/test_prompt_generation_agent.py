from magma.core.models.execution import GenerationMode, MaintenanceCategory, TemplateTask
from magma.knowledge_repository import InMemoryVectorStore, KnowledgeRepositoryAgent
from magma.knowledge_repository.embeddings import EmbeddingClient
from magma.llm_providers import MockLLMProvider
from magma.phase2_artifact_generation.prompt_generation_agent import (
    PromptGenerationAgent,
    PromptGenerationRequest,
)


def _task():
    return TemplateTask(
        template_id="tmplA", maintenance_category=MaintenanceCategory.ADDITIVE,
        generation_mode=GenerationMode.TARGETED_MODIFICATION, instructions="add retry_count param",
    )


def test_prompt_includes_task_and_template():
    llm = MockLLMProvider(responder=lambda p: "add an optional retry_count parameter, default 3")
    agent = PromptGenerationAgent(llm_provider=llm)
    result = agent.process(PromptGenerationRequest(task=_task(), template_content="def call(): ..."))
    assert result.success
    assert "retry_count" in result.output.prompt_text
    assert "def call(): ..." in result.output.prompt_text


def test_no_exemplar_recorded_when_absent():
    llm = MockLLMProvider(responder=lambda p: "intent")
    agent = PromptGenerationAgent(llm_provider=llm)  # no knowledge_repository configured
    result = agent.process(PromptGenerationRequest(task=_task(), template_content="x", exemplar_query="retry"))
    assert result.output.exemplar_absent
    assert result.output.used_exemplar_count == 0


def test_exemplars_retrieved_when_available():
    kr = KnowledgeRepositoryAgent(
        vector_store=InMemoryVectorStore(),
        embedding_client=EmbeddingClient(backend=lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts], dimensions=4),
    )
    kr.index_fragment("ex1", "retry_count example", {"content": "def call(retry_count=3): ..."})
    llm = MockLLMProvider(responder=lambda p: "intent")
    agent = PromptGenerationAgent(llm_provider=llm, knowledge_repository=kr)
    result = agent.process(PromptGenerationRequest(task=_task(), template_content="x", exemplar_query="retry_count"))
    assert not result.output.exemplar_absent
    assert result.output.used_exemplar_count == 1
    assert "retry_count=3" in result.output.prompt_text


def _task_with_metadata():
    return TemplateTask(
        template_id="tmplA", maintenance_category=MaintenanceCategory.ADDITIVE,
        generation_mode=GenerationMode.TARGETED_MODIFICATION, instructions="add retry_count param",
        generation_rules=["snake_case naming", "no wildcard imports"],
        output_formats=["python"],
        architectural_constraints="must remain stateless",
    )


def test_task_generation_rules_and_constraints_included_in_prompt():
    llm = MockLLMProvider(responder=lambda p: "intent")
    agent = PromptGenerationAgent(llm_provider=llm)
    result = agent.process(PromptGenerationRequest(task=_task_with_metadata(), template_content="x"))
    text = result.output.prompt_text
    assert "snake_case naming" in text
    assert "no wildcard imports" in text
    assert "python" in text
    assert "must remain stateless" in text


def test_annotated_template_context_pulled_from_graph_node():
    from magma.core.models.graph import GraphNode, NodeType

    kr = KnowledgeRepositoryAgent()
    kr.graph.add_node(GraphNode(
        node_type=NodeType.TEMPLATE_COMPONENT, node_id="tmplA",
        properties={"description": "generates the client stub", "language": "xtend"},
    ))
    llm = MockLLMProvider(responder=lambda p: "intent")
    agent = PromptGenerationAgent(llm_provider=llm, knowledge_repository=kr)
    result = agent.process(PromptGenerationRequest(task=_task(), template_content="x"))
    text = result.output.prompt_text
    assert "Annotated template context" in text
    assert "generates the client stub" in text


def test_exemplar_query_defaults_to_task_instructions_when_unset():
    kr = KnowledgeRepositoryAgent(
        vector_store=InMemoryVectorStore(),
        embedding_client=EmbeddingClient(backend=lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts], dimensions=4),
    )
    kr.index_fragment("ex1", "add retry_count param", {"content": "def call(retry_count=3): ..."})
    llm = MockLLMProvider(responder=lambda p: "intent")
    agent = PromptGenerationAgent(llm_provider=llm, knowledge_repository=kr)
    # No exemplar_query passed -- should fall back to task.instructions.
    result = agent.process(PromptGenerationRequest(task=_task(), template_content="x"))
    assert not result.output.exemplar_absent
    assert result.output.used_exemplar_count == 1
