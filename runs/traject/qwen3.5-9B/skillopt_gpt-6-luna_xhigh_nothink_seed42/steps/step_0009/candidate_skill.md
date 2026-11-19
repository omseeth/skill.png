# Tool Calling Skill

## Planning, tool selection, and validation
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.

- Before selecting calls, make a coverage ledger: list each requested result and its explicit constraints and facets (for example, geographic levels, time horizon, price or shipping filters, and requested metrics). Map each entry to the endpoint and schema field that provides it. Count an entry as covered only when the endpoint documents that exact result and scope; related or adjacent data do not suffice. If no single endpoint covers every entry, use separate appropriate calls. In the final audit, verify that every ledger entry is covered and every requested constraint is bound to a supported field.

- For overlapping endpoints, compare the exact input type and every requested output facet, not just the broad topic. If the user explicitly names a source or issuing authority, treat it as a constraint and prefer its matching endpoint over a different provider's similar result.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding

- For geographic fields that encode administrative hierarchies, follow the endpoint's documented hierarchy rather than treating city, province/state, metro area, and country as interchangeable. Use canonical codes where expected and provide the divisions needed to identify the requested place; use documented geographic lookup data to resolve names when available.
- When a schema splits one supplied identifier across component fields, parse it according to those field definitions and pass each component only to its corresponding field (for example, a username/local part and a domain). Do not repeat the entire compound identifier in each component field.
- Use each endpoint's exact parameter names and required/optional placement. Build the required-parameter list from fields the schema marks required, and put schema-optional fields only in the optional list—even when the user explicitly requests them. A requested value does not make an optional parameter required. Bind every constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful free text unless the schema requires a documented conversion. For categorical filters, codes, and selectors, map the request to the documented parameter meaning—not a similarly named field—and use the endpoint's canonical accepted token and casing from its enums or examples. Do not pass a descriptive word as a category value when it changes the field's meaning; preserve original wording when no conversion is documented. Interpret directional constraints according to field semantics.

- Map temporal wording to the endpoint's date-bound semantics: a start such as “since” or “from” belongs in the lower/start-bound field, while an end such as “until” or “through” belongs in the upper/end-bound field. Check the documented inclusivity and do not reverse the bounds.
- When a source or publisher filter expects API-specific identifiers, use its documented canonical form (for example, a publisher domain rather than a display name) instead of assuming the human-readable label is accepted.
- Include an optional parameter whenever it directly encodes a requested facet, even if the user phrases it in prose or the endpoint has a default. For an unfiltered, random, or full/standard result, use a documented “all” or unfiltered selector when needed to preserve that scope. Do not add options that are neither requested nor needed to express it, or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.

- If a requested or named endpoint is absent from the available-tools list, do not invent its identifier or switch to explanatory prose. Use an available endpoint only if its documented capability supports the requested result; do not use an adjacent endpoint to simulate an unsupported facet. If no available endpoint supports that result, make no call for it and still return only the required structured format.

- In this answer format, each parameter entry must be an object with exactly `name` and `value` keys; do not use shorthand objects such as `{ "limit": "10" }`. Encode every scalar `value` as a JSON string, even when the API schema declares the parameter as NUMBER or BOOLEAN. Schema types govern parameter validity, not serialization in this wrapper. Use native JSON numbers or booleans only when the response contract—not the API schema—explicitly requires them.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
