# Runtime image for the CLI entry points in magma/scripts/ -- run_workflow
# (one-shot episode), run_doc_watch (the always-on vendor-documentation
# watcher), review_queue (human review dashboard), bootstrap_schema and
# seed_knowledge_repository (Neo4j/Weaviate setup). Not an HTTP service:
# there's nothing to expose except the optional Prometheus /metrics port
# each script can open with --metrics-port.
#
# Two-stage build so build tooling (pip's resolver, compiler headers
# numpy/etc. may need) never ships in the final image; the package is
# pip-installed in the builder stage and only /usr/local is copied
# across. Installs core deps plus the "web" and "observability" groups
# explicitly (requests, prometheus-client) since both CLIs need them to
# do real work; the heavier "crewai" and "ml" groups (crewai, spacy,
# scikit-learn, xgboost) stay opt-in -- see README's "Docker" section
# for --build-arg EXTRAS.
FROM python:3.14-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY magma ./magma

ARG EXTRAS=""
RUN pip install --no-cache-dir --prefix=/install . requests prometheus-client ${EXTRAS}

FROM python:3.14-slim

RUN useradd --create-home --uid 1000 magma
COPY --from=builder /install /usr/local
RUN mkdir -p /data && chown magma:magma /data
USER magma
WORKDIR /home/magma

# Every real script takes --audit-log/--review-queue-path (or
# AUDIT_LOG_PATH/REVIEW_QUEUE_PATH); /data is the writable volume
# docker-compose.yml mounts for that state, plus review_queue.py's file.
VOLUME ["/data"]

ENTRYPOINT ["python", "-m"]
CMD ["magma.scripts.run_doc_watch", "--interval-seconds", "3600"]
