# Tool Calling Skill

## Planning, tool selection, and validation
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.

- Treat the available-tools list as a closed set: copy an exact listed tool identifier verbatim, including its endpoint suffix and punctuation; never shorten, reconstruct, or invent an endpoint. When a provider has sibling endpoints, select by documented output and schema for the requested result and constraints. For generic or customizable endpoints, inspect documented function or action selectors and use a supported matching mode rather than overlooking the endpoint because its name is broad.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding
Classify every argument from the selected tool's schema before serialization: put only fields marked required under `required parameters`, and all fields marked optional under `optional parameters`, even when the user supplied the value or the field is essential to a requested refinement. Task importance does not change schema placement; never promote an optional field to required. Bind each constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful formatting unless the schema requires a documented conversion. Follow documented formats, enums, casing, codes, and delimiters, and interpret directional constraints according to the field semantics.

- For endpoints with hierarchical location fields, map each place name or code to its actual administrative level. Keep country and parent-area identifiers in their proper fields, and include any parent levels required by a child field. Do not place a city or metro-area name in a different level merely to fill a parameter.

- When an endpoint exposes separate language fields, map each requested facet to the corresponding field rather than treating language as a single filter. For example, “in English with Turkish subtitles” calls for separate content or audio-language and subtitle-language filters.
- Include optional parameters when the request calls for them, including a documented value for an explicitly requested default behavior. Do not add unrelated optional settings or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.
