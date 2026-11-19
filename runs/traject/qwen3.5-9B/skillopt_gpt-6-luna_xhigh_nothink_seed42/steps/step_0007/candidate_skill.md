# Tool Calling Skill

## Planning, tool selection, and validation
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.

- For overlapping endpoints, compare the exact input type and every requested output facet, not just the broad topic. If the user explicitly names a source or issuing authority, treat it as a constraint and prefer its matching endpoint over a different provider's similar result.

- Match the endpoint to the requested result shape and each output facet, not merely the subject or shared input: distinguish searches from detail lookups, single items from collections, current conditions from forecasts, and generic validation from specialized checks (such as MX, disposable classification, availability, or SMTP). Use the narrowest endpoint whose documentation explicitly provides each requested result or check, splitting calls when needed. For lists, choose independently for each target and requested check; do not fan every target out across alternatives unless independent cross-checks are requested. When distinct measures require different units, prefer an endpoint with documented controls for each measure; use a generic units parameter only when its documented semantics cover them all.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding

- Treat clearly stated language, market, storefront, or region context as applying to related calls for that provider when their schemas support it. A market or region may be composite: populate each directly relevant documented field (such as country/store, currency, language/locale, or size schema) independently using a canonical value unambiguously determined by the request and documentation. Do not assume one field sets the others, propagate context to unrelated providers, infer it when absent, or invent unsupported values.
- Use each endpoint's exact parameter names and required/optional placement. Bind every constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful free text unless the schema requires a documented conversion. For categorical filters, codes, and selectors, map the request to the documented parameter meaning—not a similarly named field—and use the endpoint's canonical accepted token and casing from its enums or examples. Do not pass a descriptive word as a category value when it changes the field's meaning; preserve original wording when no conversion is documented. Interpret directional constraints according to field semantics.

- When a parameter explicitly expects a provider-specific slug or handle, use the API's canonical identifier rather than passing a display name or inventing a slug variant. Follow the provider's documented convention or lookup where available; for example, the TED speaker slug is `yuval_noah_harari`, not `yuval-noah-harari`.
- Select a date parameter according to the event the date describes. For TED Talks, use `from_record_date` for recording dates and `from_publish_date` for publication dates; an unqualified request for talks since a date follows the endpoint's recording-date convention unless publication is specified.
- Include an optional parameter whenever it directly encodes a requested facet, even if the user phrases it in prose or the endpoint has a default. For an unfiltered, random, or full/standard result, use a documented “all” or unfiltered selector when needed to preserve that scope. Do not add options that are neither requested nor needed to express it, or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.

- In this answer format, encode scalar `value` fields as JSON strings, including numeric and boolean-like parameter values, unless the task's response contract explicitly requires native JSON numbers or booleans.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
