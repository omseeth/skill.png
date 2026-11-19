# Tool Calling Skill

## Track requested actions
- Turn each explicit clause into a requested action, target, and set of qualifiers. In one-turn tasks, narrated sequences such as “first,” “then,” and “finally” describe requested actions, not actions already completed.
- Account for every requested deliverable in the calls. Use separate calls for distinct operations or targets unless an endpoint explicitly supports handling them together. Do not infer calls from incidental context or a name match, or add duplicate or alternative calls unless the user requests separate results.

## Select the right endpoint
- Choose an endpoint by its documented operation, output, target, and support for the requested qualifiers—not merely by a shared topic or entity name. Prefer the most specific endpoint that fits the request; use separate endpoints when distinct deliverables require them.
- Use the complete tool name exactly as listed, including its provider or namespace where applicable.

## Map constraints to the schema
- Carry every explicit filter and qualifier into the supported field that represents it. Follow the endpoint’s documented parameter meanings, required and optional fields, value types, canonical values, units, and encoding for multiple values. Do not assume conventions are the same across providers.
- Distinguish parameters that can be confused, such as quantities versus repetitions or operation-specific values versus ordinary search terms. Include supported optional parameters needed to express explicit constraints; do not silently relax a constrained request. Use documented defaults only when they fit the request, and do not invent unsupported arguments, identifiers, credentials, or values.

## Preserve the answer structure
- Follow the active prompt’s answer contract. Represent each invocation as its own call object with exactly the required keys and value types; keep calls in the required wrapper and do not merge calls or duplicate fields.
- Before submitting, reconcile the calls against the requested actions and qualifiers, then verify that the output is valid JSON when JSON is required, with no stray fields, nested call objects, or extra prose that violates the contract.

## List and catalog requests
- When a user asks for a list of available sources, categories, currencies, stations, or similar items, choose the endpoint that explicitly enumerates that collection. Do not substitute a general search, latest-content, or single-item detail endpoint.

<!-- SLOW_UPDATE_START -->
Before building calls, reread the prompt clause by clause and make a checklist of every requested result, target, and qualifier. Treat narrated actions as requests when the task asks you to retrieve or report them, and do not let the main topic eclipse side requests on other platforms or services. Map every checklist item to a call; every call must serve a checklist item. Make separate calls for genuinely distinct requested outputs, but do not add competing validators, generic searches, heartbeats, inbox checks, or other related calls that the user did not request.

When endpoints seem similar, consult their own tool descriptions and select by exact operation, target type, and output—not by a shared topic. Use the complete listed tool name and namespace. For named news sources, prefer a source-specific latest-news operation when it matches the request; use the source’s documented canonical identifier rather than a broad query. For email, distinguish address validation from address-level disposable checks and domain-level disposable, MX, SMTP, or role-account checks. Send an address to address endpoints and the bare domain to domain endpoints; make only the distinct checks the request calls for. For a requested fake address, use the exact fully qualified generator name, and do not submit invented placeholder addresses to inbox or verification endpoints.

Treat the endpoint schema as authoritative for every qualifier, including optional fields, enums, canonical values, and value types. Do not replace documented values with intuitive prose or omit requested limits, dates, locales, sources, or topics. For example, use the documented Amazon marketplace code (such as de, not amazon.de), pair an ASOS country with its documented language (UK maps to GB and en-GB), and use a documented Priceline search_type enum rather than guessing travel. For news searches, carry the requested result count, date boundary, topic, language, and publisher filters into their documented fields and formats.

For numeric requests, identify separately the number of objects and each object’s properties, then keep distinct actions separate. For example, five 20-sided dice require the custom-dice operation with dice=5 and sides=20; three regular dice use the regular-dice operation with dice=3; a separately requested single regular die uses the regular-die operation. A requested Ace draw needs the card operation with value=A and the documented suit constraint, not a generic draw. Likewise, include supported single-result quantities and all requested search filters even when they seem implicit.

Before submitting, compare the final calls against the checklist and tool documentation: confirm every requested result is covered, no call is unsupported or redundant, each value uses the documented field and type, and each tool name exactly matches the catalog.
<!-- SLOW_UPDATE_END -->
