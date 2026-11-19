# Tool Calling Skill

## Track requested actions
- Turn each explicit clause into a requested action, target, and set of qualifiers. In one-turn tasks, narrated sequences such as “first,” “then,” and “finally” describe requested actions, not actions already completed.
- Account for every requested deliverable in the calls. Use separate calls for distinct operations or targets unless an endpoint explicitly supports handling them together. Do not infer calls from incidental context or a name match, or add duplicate or alternative calls unless the user requests separate results.

## Select the right endpoint
- Choose an endpoint by its documented operation, output, target, and support for the requested qualifiers—not merely by a shared topic or entity name. Treat the requested operation as a hard requirement: autocomplete returns suggestions, not product listings, and a current-weather endpoint does not provide a forecast unless its documentation says so. For distinct requested outputs, select the endpoint documented for each operation even when the target or topic is the same. For hierarchical targets, match the exact level and its corresponding identifier field. If a requested source has no endpoint named for it, inspect the full tool list for an endpoint that provides the requested content filtered by that source. Prefer the most specific endpoint that fits the request; use separate endpoints when distinct deliverables require them.
- Use the complete tool name exactly as listed, including its provider or namespace where applicable.

## Map constraints to the schema

- When an endpoint provides dedicated fields for explicit criteria such as source or publisher, sentiment, content type, date range, sort order, or result count, map each criterion to its field rather than expressing it only in free-text search terms. Keep the subject in the search query and place filters in their documented fields and parameter buckets.
- Interpret continuation requests such as “more,” “next,” or “continue” relative to earlier results. When the endpoint supports pagination, advance the page for the later result set and keep that page attached to its requested filters or sort order; do not repeat the first page.

- If a documented default already guarantees a requested singular result count, rely on it rather than adding a count or amount parameter solely to restate that count.
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
