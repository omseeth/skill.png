# Tool Calling Skill

## Track requested actions
- Turn each explicit clause into a requested action, target, and set of qualifiers. In one-turn tasks, narrated sequences such as “first,” “then,” and “finally” describe requested actions, not actions already completed.
- Account for every requested deliverable in the calls. Use separate calls for distinct operations or targets unless an endpoint explicitly supports handling them together. Do not infer calls from incidental context or a name match, or add duplicate or alternative calls unless the user requests separate results.

## Select the right endpoint

- Treat a named platform or source as a constraint: do not substitute a similarly named endpoint from another service. When a request supplies a stable record identifier and asks for that item’s details, use the matching source’s detail or lookup endpoint keyed by that identifier, not a keyword-search endpoint.
- For a clause that requests multiple checks or operations on one target, identify each requested outcome before selecting endpoints. Use one endpoint only when its documented output covers the full requested outcome; otherwise, use the best-fitting endpoint for each distinct operation or check. Do not treat a broad or narrower partial match as coverage of the whole clause.
- Choose an endpoint by its documented operation, output, target, and support for the requested qualifiers—not merely by a shared topic or entity name. Prefer the most specific endpoint that fits the request; use separate endpoints when distinct deliverables require them.
- Use the complete tool name exactly as listed, including its provider or namespace where applicable.

## Map constraints to the schema

- For open-ended sampling from a stated complete or standard set, check whether omitting selector fields would narrow the results; use documented unrestricted values when needed to preserve the requested scope. Map sort phrases to the enum with the same meaning—best match or relevance is not best seller or popularity—and do not substitute a nearby mode.

- Treat result count and page position as distinct constraints. Map counts such as “one” or “first N results” to the supported limit/page-size field, and map requests for page 1 or the first page to the page field when available. Supply both when the request specifies both.
- Carry every explicit filter and qualifier into the supported field that represents it. Follow the endpoint’s documented parameter meanings, required and optional fields, value types, canonical values, units, and encoding for multiple values. Do not assume conventions are the same across providers.
- Distinguish parameters that can be confused, such as quantities versus repetitions or operation-specific values versus ordinary search terms. Include supported optional parameters needed to express explicit constraints; do not silently relax a constrained request. Use documented defaults only when they fit the request, and do not invent unsupported arguments, identifiers, credentials, or values.

## Preserve the answer structure
- Follow the active prompt’s answer contract. Represent each invocation as its own call object with exactly the required keys and value types; keep calls in the required wrapper and do not merge calls or duplicate fields.
- Before submitting, reconcile the calls against the requested actions and qualifiers, then verify that the output is valid JSON when JSON is required, with no stray fields, nested call objects, or extra prose that violates the contract.

## List and catalog requests
- When a user asks for a list of available sources, categories, currencies, stations, or similar items, choose the endpoint that explicitly enumerates that collection. Do not substitute a general search, latest-content, or single-item detail endpoint.

<!-- SLOW_UPDATE_START -->
For a long compound or workflow-reconstruction request, make a private numbered ledger of every distinct requested result, target, and qualifier (including dates, pages, and units) before composing calls. Map every ledger row to the exact endpoint and call or calls it needs; then audit in both directions so every requested result is covered and no call lacks a requested purpose. Do not stop after handling only the most obvious cluster of actions.

When endpoints overlap, compare their documented operation and parameter schema to the requested output and choose the best-fitting endpoint; a shared provider, topic, or entity name is not enough. Use one best-fitting call per action-target unless the endpoint explicitly batches them or a distinct requested deliverable needs another operation. In email checks, pass an email address to an address-validation endpoint and a bare domain to a domain-disposable endpoint; do not pass a domain as an email or fan out across competing validators for the same check.

Classify parameters as required or optional according to the endpoint schema, not according to how strongly the user stated a constraint. Include every supported explicit constraint in its proper bucket. Use the schema’s canonical values, encoding, and types—for example, a market code rather than a hostname when a code is expected, and the correct date field and format for each flight lookup. Keep each qualifier attached to the operation and target it describes.

Before submitting, reconcile the ledger against the calls, checking endpoint identity, parameter names and buckets, values and types, and duplicate calls. Then verify that the complete response parses as valid JSON in the active prompt’s exact wrapper and object shape.
<!-- SLOW_UPDATE_END -->
