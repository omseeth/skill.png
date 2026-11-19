# Tool Calling Skill

## Track requested actions
- Turn each explicit clause into a requested action, target, and set of qualifiers. In one-turn tasks, treat imperative sequences and narrative or retrospective sequences (for example, “the user first accessed ... then retrieved ...”) as requested actions unless the prompt explicitly says they were completed outside this task. Before choosing tools, make an internal ledger in query order with one row per requested operation and result; expand listed targets and sources, including those in later or final clauses. Match every row to a call and every call to a row, without treating related searches as coverage of distinct results. Group rows into one call only when the endpoint explicitly supports doing so.
- Account for every requested deliverable in the calls. Use separate calls for distinct operations or targets unless an endpoint explicitly supports handling them together. Do not infer calls from incidental context or a name match, or add duplicate or alternative calls unless the user requests separate results.

## Select the right endpoint
- Choose an endpoint by its documented operation, output, target, and support for the requested qualifiers—not merely by a shared topic or entity name. Prefer the most specific endpoint that fits the request; use separate endpoints when distinct deliverables require them.

- When endpoints overlap, classify the requested result and operation, then compare them with the specific endpoint’s documented output and fields—not provider-level marketing text or a shared topic or entity name. A deliverable is covered only if that endpoint returns the requested result with its qualifiers (for example, people versus events, verification versus a related risk check, or a named-source feed versus a general topic search). Match cardinality as well: use an endpoint that supports the requested number of results and provide its quantity field when available. Every emitted call must map to an explicit requested outcome.
- Use the complete tool name exactly as listed, including its provider or namespace where applicable.

## Map constraints to the schema
- Carry every explicit filter and qualifier into the supported field that represents it. Follow the endpoint’s documented parameter meanings, required and optional fields, value types, canonical values, units, and encoding for multiple values. Do not assume conventions are the same across providers.

- Bind each qualifier only to the operation it modifies, and include it only on endpoints whose schemas support it. For hierarchical filters, map each level to its documented field and honor stated dependencies rather than substituting one level for another. Before serialization, audit each call against its schema: include every required parameter as a correctly formed name/value pair, put optional parameters in the designated optional slot when the contract provides one, avoid bare names or placeholders, and preserve declared value types and exact enum or canonical values.
- Distinguish parameters that can be confused, such as quantities versus repetitions or operation-specific values versus ordinary search terms. Include supported optional parameters needed to express explicit constraints; do not silently relax a constrained request. Use documented defaults only when they fit the request, and do not invent unsupported arguments, identifiers, credentials, or values.

## Preserve the answer structure
- Follow the active prompt’s answer contract. Represent each invocation as its own call object with exactly the required keys and value types; keep calls in the required wrapper and do not merge calls or duplicate fields.
- Before submitting, reconcile the calls against the requested actions and qualifiers, then verify that the output is valid JSON when JSON is required, with no stray fields, nested call objects, or extra prose that violates the contract.

## List and catalog requests
- When a user asks for a list of available sources, categories, currencies, stations, or similar items, choose the endpoint that explicitly enumerates that collection. Do not substitute a general search, latest-content, or single-item detail endpoint.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
