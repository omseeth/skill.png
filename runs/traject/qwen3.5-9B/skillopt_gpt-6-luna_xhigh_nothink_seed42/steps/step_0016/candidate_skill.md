# Tool Calling Skill

## Planning, tool selection, and validation
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.

- Before selecting calls, make a coverage ledger: list each requested result and its explicit constraints and facets (for example, geographic levels, time horizon, price or shipping filters, and requested metrics). Map each entry to the endpoint and schema field that provides it. Count an entry as covered only when the endpoint documents that exact result and scope; related or adjacent data do not suffice. If no single endpoint covers every entry, use separate appropriate calls. In the final audit, verify that every ledger entry is covered and every requested constraint is bound to a supported field.

- For overlapping endpoints, compare the exact input type and every requested output facet, not just the broad topic. If the user explicitly names a source or issuing authority, treat it as a constraint and prefer its matching endpoint over a different provider's similar result.

- Require an endpoint to support every explicit scope dimension: for example, a request for a news category and country needs an endpoint with both category and region filters, not a category-only feed. Treat separately requested current conditions and a time-bounded forecast as distinct results; use a dedicated forecast endpoint when available, bind the requested horizon there, and do not count a current-data endpoint as covering the forecast merely because its description mentions forecasts.
- If candidates still overlap, prefer the endpoint whose documented operation and input mode are most narrowly aligned with the request. Use a generic, custom, or broader endpoint only when its added capability is requested.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding

- For geographic fields that encode administrative hierarchies, follow the endpoint's documented hierarchy rather than treating city, province/state, metro area, and country as interchangeable. Use canonical codes where expected and provide the divisions needed to identify the requested place; use documented geographic lookup data to resolve names when available.

- For location-specific weather alerts, use a supported `point` filter for a named locality or area; use a state filter only when the requested scope is state-wide. Supply coordinates from the request or resolve them with a reliable geographic lookup rather than substituting the broader state or inventing coordinates.
- When a schema splits one supplied identifier across component fields, parse it according to those field definitions and pass each component only to its corresponding field (for example, a username/local part and a domain). Do not repeat the entire compound identifier in each component field.
- Use each endpoint's exact parameter names and required/optional placement. Build the required-parameter list from fields the schema marks required, and put schema-optional fields only in the optional list—even when the user explicitly requests them. A requested value does not make an optional parameter required. Bind every constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful free text unless the schema requires a documented conversion. For categorical filters, codes, and selectors, map the request to the documented parameter meaning—not a similarly named field—and use the endpoint's canonical accepted token and casing from its enums or examples. Do not pass a descriptive word as a category value when it changes the field's meaning; preserve original wording when no conversion is documented. Interpret directional constraints according to field semantics.
- Include an optional parameter whenever it directly encodes a requested facet, even if the user phrases it in prose or the endpoint has a default. For an unfiltered, random, or full/standard result, use a documented “all” or unfiltered selector when needed to preserve that scope. Do not add options that are neither requested nor needed to express it, or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

- If an endpoint exposes an optional `Content-Type` parameter for a JSON request, set it to `application/json` unless its documentation specifies another media type.

- As a narrow exception, include an optional parameter when the endpoint explicitly recommends supplying it for successful or faster results and its value can be safely derived from context. For example, when a ski-snow endpoint recommends an explicit unit selector, use the documented customary unit for the named location if the user has not specified a preference; do not guess among unsupported values.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.

- In this answer format, encode scalar `value` fields as JSON strings, including numeric and boolean-like parameter values, unless the task's response contract explicitly requires native JSON numbers or booleans.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.

<!-- SLOW_UPDATE_START -->
When a request recounts a workflow, carry out every enumerated operation unless the user clearly asks only for acknowledgment; check the final call list against that operation checklist.

For each requested outcome, match the endpoint to the exact operation and input type, not just the topic or provider name. For email tasks, distinguish address-format validation, deliverability or SMTP verification, disposable-email checks, disposable-domain checks, role-account checks, MX records, domain availability, and documentation; use the endpoint that supports the requested check and pass an address or domain in the form its schema expects. For product or game research, treat ratings and reviews as separate results from deals, population, or store lists, and include a dedicated supported endpoint when reviews are requested.

Before finalizing search and listing calls, inspect their documented optional fields for constraints and relevant market context, such as locale, store, currency, page, page size, limit, sort, or exact-match selectors. Bind stated constraints to those fields; use contextual values or defaults only when documented and unambiguous. Do not omit a requested facet or add unsupported guesses.

Disambiguate identically or similarly named tools by their full provider-qualified identifiers and documented capability. Keep the call list limited to operations relevant to the user’s request; do not add unrelated catalog tools merely because they share a broad topic.

As a final serialization pass, verify each field’s required-versus-optional placement and canonical token casing, then encode every scalar parameter value as a JSON string—including numbers and boolean-like values—unless the response contract explicitly requires a native type.
<!-- SLOW_UPDATE_END -->
