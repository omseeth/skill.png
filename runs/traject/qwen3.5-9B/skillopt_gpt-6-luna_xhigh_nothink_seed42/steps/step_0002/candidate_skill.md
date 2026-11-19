# Tool Calling Skill

## Planning, tool selection, and validation

- Maintain a coverage checklist of distinct requested outputs and targets. Split compound requests by independently requested data, search terms, or time horizons. Count an item as covered only when the endpoint's documented output and supported filters meet that exact request. Use the minimum relevant calls to cover the checklist; do not substitute a merely related endpoint or add alternate tools that repeat the same check.
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding
- Use each endpoint's exact parameter names and required/optional placement. Bind every constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful formatting unless the schema requires a documented conversion. Follow documented formats, enums, casing, codes, and delimiters, and interpret directional constraints according to the field semantics.

- Translate status qualifiers into their explicit documented filters even when the endpoint can return broader results; for example, a request for active alerts calls for an `active=1` filter when supported.

- Map each natural-language refinement to the matching documented parameter even when the user does not name that field. Use documented codes or values unambiguously derivable from the query; normalize place names only when the endpoint's documented code format requires it.
- Include optional parameters when the request calls for them, including a documented value for an explicitly requested default behavior. Do not add unrelated optional settings or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.
- Return a syntactically valid JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Every entry in either parameter list must be an object shaped as `{\"name\": \"<exact schema parameter>\", \"value\": <value>}`. Check that braces, brackets, commas, and quotes are valid, and include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.
