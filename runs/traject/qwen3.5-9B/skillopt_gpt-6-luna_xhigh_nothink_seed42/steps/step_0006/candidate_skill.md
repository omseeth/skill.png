# Tool Calling Skill

## Planning, tool selection, and validation
- Treat every explicitly described action and requested detail as part of the task, including steps narrated as background or a sequence; count an operation as already completed only when the prompt supplies its result or explicitly says not to repeat it. Make a checklist with one entry per desired result and its requested facets, and verify that each is covered by an appropriate call.
- For each checklist item, shortlist only endpoints whose documented capability returns the requested kind of result for that resource or category. Use the exact full tool identifier from the available-tools list and follow its documented schema. Choose the single best-supported endpoint unless the user explicitly requests multiple sources or distinct facets require separate calls. Do not substitute tangential endpoints, fan an item out across unrelated catalogs, treat suggestions as product listings, infer capabilities from a provider name, or invent an endpoint absent from the available-tools list.

- For overlapping endpoints, compare the exact input type and every requested output facet, not just the broad topic. If the user explicitly names a source or issuing authority, treat it as a constraint and prefer its matching endpoint over a different provider's similar result.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding
- For each call, map every requested constraint and output facet independently to its documented parameter; do not assume one filter implies another. Bind each constraint only to the call it modifies, and do not carry dates, limits, filters, or pagination across operations. Put each parameter in the required or optional list exactly as the schema classifies it: a requested filter remains optional when the endpoint marks it optional. Distinguish regional storefront hostnames from marketplace codes: if the field expects a country or region code, derive it from the supplied storefront or region; if it expects a hostname, preserve the hostname.
- Preserve user-provided identifiers and meaningful free text unless the schema requires a documented conversion. For categorical filters, codes, and selectors, map the request to the documented parameter meaning—not a similarly named field—and use the endpoint's canonical accepted token and casing from its enums or examples. Do not pass a descriptive word as a category value when it changes the field's meaning; preserve original wording when no conversion is documented. Interpret directional constraints according to field semantics.
- Include an optional parameter whenever it directly encodes a requested facet, even if the user phrases it in prose or the endpoint has a default. For the first N results from a paginated endpoint, set page to 1 and the supported limit or per-page parameter to N, even when page 1 is implied rather than stated. For an unfiltered, random, or full/standard result, use a documented “all” or unfiltered selector when needed to preserve that scope. Do not add options that are neither requested nor needed to express it, or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

## Final response check

- If the response contract specifies an ordering (for example, sorting calls by tool name), reorder the call objects to match it after selecting and validating the calls; do not simply preserve query order.
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.

- In this answer format, encode scalar `value` fields as JSON strings, including numeric and boolean-like parameter values, unless the task's response contract explicitly requires native JSON numbers or booleans.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
