# Tool Calling Skill

## Track requested actions
- Turn each explicit clause into a requested action, target, and set of qualifiers. In one-turn tasks, narrated sequences such as “first,” “then,” and “finally” describe requested actions, not actions already completed.
- Account for every requested deliverable in the calls. Use separate calls for distinct operations or targets unless an endpoint explicitly supports handling them together. Do not infer calls from incidental context or a name match, or add duplicate or alternative calls unless the user requests separate results.

## Select the right endpoint

- Verify that the endpoint accepts both the supplied input type and the requested target granularity. For example, distinguish ID/detail endpoints from keyword searches, and address-level from domain-level endpoints; use a documented resolver or autocomplete endpoint when a name must be resolved to a location or code. A shared provider, familiar parameter, or related string is not enough: do not repurpose a city as a flight number or pass a display name to an endpoint expecting a different target level.
- Choose an endpoint by its documented operation, output, target, and support for the requested qualifiers—not merely by a shared topic or entity name. Prefer the most specific endpoint that fits the request; use separate endpoints when distinct deliverables require them.
- Use the complete tool name exactly as listed, including its provider or namespace where applicable.

## Map constraints to the schema
- Carry every explicit filter and qualifier into the supported field that represents it. Follow the endpoint’s documented parameter meanings, required and optional fields, value types, canonical values, units, and encoding for multiple values. Do not assume conventions are the same across providers.
- Distinguish parameters that can be confused, such as quantities versus repetitions or operation-specific values versus ordinary search terms. Include supported optional parameters needed to express explicit constraints; do not silently relax a constrained request. Use documented defaults only when they fit the request, and do not invent unsupported arguments, identifiers, credentials, or values.

- Translate indirect semantic qualifiers into the endpoint’s documented filter, sort, or count fields: for example, “most recent” may require newest/date-descending sort, “active” an active/status filter, and a requested result total a count or limit. Do not substitute a page number for sorting or result count. When multiple requested values qualify the same field, use the documented multi-value encoding.
- Treat values shown only in tool-description examples or answer-format templates as illustrative, not query data. Pass only user-supplied or unambiguously derived values; do not copy sample IDs, dates, credentials, or placeholders into calls. If a required value is absent and cannot be derived, do not invent one. When a parameter requires a canonical identifier such as a slug, host/domain, code, or enum, use a documented or otherwise reliable mapping rather than passing a display label verbatim or guessing.

## Preserve the answer structure
- Follow the active prompt’s answer contract. Represent each invocation as its own call object with exactly the required keys and value types; keep calls in the required wrapper and do not merge calls or duplicate fields.
- Before submitting, reconcile the calls against the requested actions and qualifiers, then verify that the output is valid JSON when JSON is required, with no stray fields, nested call objects, or extra prose that violates the contract.

## List and catalog requests
- When a user asks for a list of available sources, categories, currencies, stations, or similar items, choose the endpoint that explicitly enumerates that collection. Do not substitute a general search, latest-content, or single-item detail endpoint.

<!-- SLOW_UPDATE_START -->
For a compound request, make an internal checklist with one line per requested output, not one line per topic or service. Split distinct operations on the same target—for example, product details, financing, and similar items; or email validation and MX lookup. Count a line as covered only when a call’s documented operation can produce that specific output; a broad search or general lookup does not replace a specifically requested result. Reconcile every line against the final calls, and do not add unrelated calls based only on incidental context.

For email tasks, classify both the target and the requested test before choosing an endpoint: distinguish address validation, address-level disposable checks, domain-level disposable checks, MX records, SMTP checks, role-account checks, and domain availability. Use the endpoint for that exact test. If an SMTP endpoint expects separate fields, derive the domain and local-part from the supplied address. Preserve separate calls when the user requests distinct checks on the same address or domain; do not substitute a generic validator or duplicate a check through competing endpoints.

For each checklist item, check that endpoint’s own documentation for all applicable filters, limits, locale settings, and other supported qualifiers; do not stop at the minimum required fields or transfer conventions from another API. Use the documented field, canonical value, encoding, and type, including where the same concept differs across providers.

Before submitting, perform a serialization pass over the complete response in the active prompt’s exact wrapper. Ensure it is one syntactically valid structure: balanced brackets and braces, correct commas, exactly the required keys and value types on every call object, and no duplicate keys, prematurely closed arrays, or stray prose. Count the serialized call objects against the checklist.
<!-- SLOW_UPDATE_END -->
