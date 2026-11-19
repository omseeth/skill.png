# Tool Calling Skill

## Track requested actions
- Turn each explicit clause into a requested action, target, and set of qualifiers. In one-turn tasks, narrated sequences such as “first,” “then,” and “finally” describe requested actions, not actions already completed.
- Account for every requested deliverable in the calls. Use separate calls for distinct operations or targets unless an endpoint explicitly supports handling them together. Do not infer calls from incidental context or a name match, or add duplicate or alternative calls unless the user requests separate results.

## Select the right endpoint
- Choose an endpoint by its documented operation, output, target, and support for the requested qualifiers—not merely by a shared topic or entity name. Prefer the most specific endpoint that fits the request; use separate endpoints when distinct deliverables require them.

- Treat an explicitly named provider, endpoint, or source as part of the target, and choose the matching documented endpoint rather than a generic or neighboring one. Distinguish adjacent operations and scopes, and match each requested category to an endpoint with the corresponding output and scope (for example, a named publisher's feed versus general headlines in that language). If the user asks to use multiple tools or methods, identify the distinct checks or evidence requested and choose a best-fit endpoint for each; do not call every related endpoint or repeat a call. If one endpoint's documented output covers multiple requested aspects, do not add an overlapping call or alternate search unless the user explicitly asks for independent sources or checks.
- Use the complete tool name exactly as listed, including its provider or namespace where applicable.

## Map constraints to the schema
- Carry every explicit filter and qualifier into the supported field that represents it. Follow the endpoint’s documented parameter meanings, required and optional fields, value types, canonical values, units, and encoding for multiple values. Do not assume conventions are the same across providers.
- Distinguish parameters that can be confused, such as quantities versus repetitions or operation-specific values versus ordinary search terms. Include supported optional parameters needed to express explicit constraints; do not silently relax a constrained request. Use documented defaults only when they fit the request, and do not invent unsupported arguments, identifiers, credentials, or values.

## Preserve the answer structure
- Follow the active prompt’s answer contract. Represent each invocation as its own call object with exactly the required keys and value types; keep calls in the required wrapper and do not merge calls or duplicate fields.
- Before submitting, reconcile the calls against the requested actions and qualifiers, then serialize strictly to the active contract: produce one complete JSON value in the specified wrapper, use the exact call-object keys and parameter-entry shapes (for example, `{ "name": ..., "value": ... }` where required), and put arguments in the declared required or optional collection. Check balanced delimiters and valid JSON; include no explanation, metadata, or trailing content unless explicitly allowed.

## List and catalog requests
- When a user asks for a list of available sources, categories, currencies, stations, or similar items, choose the endpoint that explicitly enumerates that collection. Do not substitute a general search, latest-content, or single-item detail endpoint.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
