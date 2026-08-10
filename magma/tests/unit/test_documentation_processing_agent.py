from magma.core.models.documentation import FragmentLabel
from magma.knowledge_repository import KnowledgeRepositoryAgent
from magma.phase1_documentation_detection import DocumentationProcessingAgent, RawDocument


def test_labels_deprecation_notice():
    agent = DocumentationProcessingAgent()
    doc = RawDocument(
        content="# Notes\n\nThe old_field is deprecated and will be removed in v3.\n",
        source_url="https://example.com", vendor="Acme", version="v1",
    )
    result = agent.process(doc)
    assert result.success
    assert any(f.label == FragmentLabel.DEPRECATION_NOTICE for f in result.output)


def test_writes_fragments_to_knowledge_repository():
    kr = KnowledgeRepositoryAgent()
    agent = DocumentationProcessingAgent(knowledge_repository=kr)
    doc = RawDocument(content="# T\n\nsome content\n", source_url="u", vendor="V", version="1")
    result = agent.process(doc)
    assert result.success
    assert len(kr.graph._nodes) == len(result.output)


def test_skips_empty_source():
    agent = DocumentationProcessingAgent()
    result = agent.process(RawDocument(content="   \n", source_url="u", vendor="V", version="1"))
    assert not result.success
    assert len(agent.skipped) == 1
