"""Stage 2 of the Documentation Processing Agent: NER, dependency parsing,
and topic modeling.

Uses a fine-tuned transformer pipeline (spaCy) rather than LLM prompting,
for reproducibility and throughput. `SpacyNLPBackend` wraps that
production pipeline; `HeuristicNLPBackend` is
a dependency-light fallback (regex-based entity extraction) used in tests
and environments where the spaCy transformer model is not downloaded, so
the rest of the agent is exercised without a large model dependency.
"""
from __future__ import annotations

import abc
import re
from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    text: str
    label: str
    start: int
    end: int


@dataclass
class ExtractedRelation:
    head: str
    relation: str
    dependent: str


@dataclass
class NLPAnnotation:
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    topics: list[str]


class NLPBackend(abc.ABC):
    @abc.abstractmethod
    def annotate(self, text: str) -> NLPAnnotation: ...


# Heuristic patterns approximating the entity classes we care about:
# API names, parameter types, version numbers.
_API_NAME_RE = re.compile(r"\b([a-z_][a-zA-Z0-9_]*\([^)]*\)|[A-Z][a-zA-Z0-9]*(?:\.[A-Z][a-zA-Z0-9]*)+)\b")
_PARAM_RE = re.compile(r"\b([a-z_][a-zA-Z0-9_]*)\s*(?::|=)\s*(int|str|string|bool|float|list|dict|optional)\b", re.IGNORECASE)
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:\.[a-zA-Z0-9]+)?\b")
_REQUIREMENT_VERB_RE = re.compile(r"\b(requires?|must|deprecat\w+|remov\w+|throws?|raises?)\b", re.IGNORECASE)


class HeuristicNLPBackend(NLPBackend):
    """Regex-based fallback: no external model, fully deterministic."""

    def annotate(self, text: str) -> NLPAnnotation:
        entities: list[ExtractedEntity] = []
        for match in _API_NAME_RE.finditer(text):
            entities.append(ExtractedEntity(text=match.group(1), label="API_NAME", start=match.start(), end=match.end()))
        for match in _PARAM_RE.finditer(text):
            entities.append(ExtractedEntity(text=match.group(1), label="PARAMETER_TYPE", start=match.start(), end=match.end()))
        for match in _VERSION_RE.finditer(text):
            entities.append(ExtractedEntity(text=match.group(0), label="VERSION", start=match.start(), end=match.end()))

        relations: list[ExtractedRelation] = []
        for verb_match in _REQUIREMENT_VERB_RE.finditer(text):
            verb = verb_match.group(1).lower()
            nearby = [e for e in entities if abs(e.start - verb_match.start()) < 80]
            for e in nearby:
                relations.append(ExtractedRelation(head=verb, relation="applies_to", dependent=e.text))

        topics = sorted({e.label.lower() for e in entities}) or ["general"]
        return NLPAnnotation(entities=entities, relations=relations, topics=topics)


class SpacyNLPBackend(NLPBackend):
    """Production backend: spaCy transformer pipeline.

    `spacy` and the `en_core_web_trf` model are imported lazily; this class
    only needs them at construction time, keeping the module importable in
    environments where the model has not been downloaded.
    """

    def __init__(self, model_name: str = "en_core_web_trf") -> None:
        import spacy  # noqa: F401

        self._nlp = spacy.load(model_name)

    def annotate(self, text: str) -> NLPAnnotation:
        doc = self._nlp(text)
        entities = [ExtractedEntity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char) for ent in doc.ents]
        relations = [
            ExtractedRelation(head=token.head.text, relation=token.dep_, dependent=token.text)
            for token in doc
            if token.dep_ not in ("punct", "ROOT")
        ]
        topics = self._topic_model(doc)
        return NLPAnnotation(entities=entities, relations=relations, topics=topics)

    @staticmethod
    def _topic_model(doc) -> list[str]:
        # A full LDA/NMF topic model (scikit-learn) is fit over
        # the fragment corpus at the pipeline level, not per-call; here we
        # fall back to salient noun chunks as a lightweight per-fragment
        # topic signal.
        return sorted({chunk.root.lemma_.lower() for chunk in doc.noun_chunks})[:5]
