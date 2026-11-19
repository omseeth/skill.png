# Tool Calling Skill

## Planning, tool selection, and validation
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding
- Use each endpoint's exact parameter names and required/optional placement. Bind every constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful formatting unless the schema requires a documented conversion. Follow documented formats, enums, casing, codes, and delimiters, and interpret directional constraints according to the field semantics.

- When binding a search, query, or criteria parameter, carry through every material qualifier in the request (such as color, material, audience, or intent). Do not silently broaden the search or reduce it to a generic term; normalize wording only as needed by the endpoint without dropping the requested scope.

- For multi-valued filters with a separate match-mode parameter, encode all requested values using the endpoint's collection format and set the mode to match the user's conjunction or disjunction. Keep filter values separate from the match-mode value.
- Include optional parameters when the request calls for them, including a documented value for an explicitly requested default behavior. Do not add unrelated optional settings or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.

- When the output contract specifies an ordering, sort call objects by the exact full tool identifier before serialization rather than retaining the query's narrative order.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.

- When an endpoint explicitly exposes a `Content-Type` parameter for a JSON request, include its JSON media type (for example, `application/json`) even if the user does not mention the transport format; treat it as endpoint configuration, not an unrelated filter.
