# Tool Calling Skill

## Track requested actions
- Turn each explicit clause into a requested action, target, and set of qualifiers. In one-turn tasks, narrated sequences such as “first,” “then,” and “finally” describe requested actions, not actions already completed.
- Account for every requested deliverable in the calls. Use separate calls for distinct operations or targets unless an endpoint explicitly supports handling them together. Do not infer calls from incidental context or a name match, or add duplicate or alternative calls unless the user requests separate results.

## Select the right endpoint
- Choose an endpoint by its documented operation, output, target, and support for the requested qualifiers—not merely by a shared topic or entity name. Prefer the most specific endpoint that fits the request; use separate endpoints when distinct deliverables require them.
- Use the complete tool name exactly as listed, including its provider or namespace where applicable.

## Map constraints to the schema

- Map requested result cardinality to documented count or amount fields: when the user requests one item and the endpoint exposes a parameter controlling how many items it returns, pass `1`; for an explicit larger total, pass that total. Do not confuse result count with length, pages, or repetitions, and do not invent a count parameter when none is supported.
- Carry every explicit filter and qualifier into the supported field that represents it. Follow the endpoint’s documented parameter meanings, required and optional fields, value types, canonical values, units, and encoding for multiple values. Do not assume conventions are the same across providers.
- Distinguish parameters that can be confused, such as quantities versus repetitions or operation-specific values versus ordinary search terms. Include supported optional parameters needed to express explicit constraints; do not silently relax a constrained request. Use documented defaults only when they fit the request, and do not invent unsupported arguments, identifiers, credentials, or values.

## Preserve the answer structure
- Follow the active prompt’s answer contract. Represent each invocation as its own call object with exactly the required keys and value types; keep calls in the required wrapper and do not merge calls or duplicate fields.

- Distinguish an endpoint’s native parameter type from the active call format’s serialized `value` type. When the contract represents parameters as `{"name": ..., "value": ...}` and requires `value` to be a string, serialize numeric and boolean inputs as canonical strings (for example, `"3"` or `"true"`), not JSON numbers or booleans. Follow any explicit contrary wrapper requirement.
- Before submitting, reconcile the calls against the requested actions and qualifiers, then verify that the output is valid JSON when JSON is required, with no stray fields, nested call objects, or extra prose that violates the contract.

## List and catalog requests
- When a user asks for a list of available sources, categories, currencies, stations, or similar items, choose the endpoint that explicitly enumerates that collection. Do not substitute a general search, latest-content, or single-item detail endpoint.

<!-- SLOW_UPDATE_START -->
For a compound request, make an internal checklist with one line per requested output, not one line per topic or service. Split distinct operations on the same target—for example, product details, financing, and similar items; or email validation and MX lookup. Count a line as covered only when a call’s documented operation can produce that specific output; a broad search or general lookup does not replace a specifically requested result. Reconcile every line against the final calls, and do not add unrelated calls based only on incidental context.

For email tasks, classify both the target and the requested test before choosing an endpoint: distinguish address validation, address-level disposable checks, domain-level disposable checks, MX records, SMTP checks, role-account checks, and domain availability. Use the endpoint for that exact test. If an SMTP endpoint expects separate fields, derive the domain and local-part from the supplied address. Preserve separate calls when the user requests distinct checks on the same address or domain; do not substitute a generic validator or duplicate a check through competing endpoints.

For each checklist item, check that endpoint’s own documentation for all applicable filters, limits, locale settings, and other supported qualifiers; do not stop at the minimum required fields or transfer conventions from another API. Use the documented field, canonical value, encoding, and type, including where the same concept differs across providers.

Before submitting, perform a serialization pass over the complete response in the active prompt’s exact wrapper. Ensure it is one syntactically valid structure: balanced brackets and braces, correct commas, exactly the required keys and value types on every call object, and no duplicate keys, prematurely closed arrays, or stray prose. Count the serialized call objects against the checklist.
<!-- SLOW_UPDATE_END -->
