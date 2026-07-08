# Security Policy

## Supported Versions

MAGMA is pre-1.0 (`0.x`); only the latest commit on `main` is supported. There is no
long-term-support branch yet.

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting (Security tab -> "Report a
vulnerability") on this repository, or email elenaakik11@gmail.com with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal repro is ideal)
- The affected file(s)/module(s), if known

You should get an acknowledgment within a few days. We'll work with you to understand and
fix the issue before any public disclosure.

## Scope notes specific to this project

A few things worth knowing when evaluating MAGMA's attack surface:

- **Generated code execution**: Phase 3's default `FunctionalValidator` executes generated
  artifacts in a subprocess with a timeout (`magma/phase3_validation/validators.py`), not a
  fully isolated sandbox (no memory/CPU rlimits, no network namespace). Don't point MAGMA at
  untrusted documentation sources you don't control if code execution isolation is a hard
  requirement for your deployment -- treat it the way you'd treat any CI job that runs
  generated code.
- **LLM API keys / credentials**: `.env.example` lists every credential MAGMA reads
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `NEO4J_PASSWORD`, `WEAVIATE_API_KEY`). None of these
  are logged; `AuditLogger`/`JsonFileTicketStore` persist agent actions and review tickets, not
  credentials.
- **Vendor documentation fetching**: `HttpDocumentFetcher` fetches and parses HTML/Markdown
  from vendor URLs you configure in `config/vendor_sources.yaml` -- it strips `<script>`/
  `<style>` content but is not a hardened HTML sanitizer; the resulting text is fed to an LLM,
  not rendered or executed.
