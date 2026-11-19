# Tool Calling Skill

## Planning, tool selection, and validation

- A provider mentioned only as part of a past-workflow narrative does not by itself restrict provider selection. Unless the user explicitly requires that source, inspect the available tools for relevant supported coverage across providers. Use an alternative only for the facets its documentation supports; do not imply that it provides unsupported filters, time ranges, or categories.
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.

- Before selecting calls, make a coverage ledger: list each requested result and its explicit constraints and facets (for example, geographic levels, time horizon, price or shipping filters, and requested metrics). Map each entry to the endpoint and schema field that provides it. Count an entry as covered only when the endpoint documents that exact result and scope; related or adjacent data do not suffice. If no single endpoint covers every entry, use separate appropriate calls. In the final audit, verify that every ledger entry is covered and every requested constraint is bound to a supported field.

- For overlapping endpoints, compare the exact input type and every requested output facet, not just the broad topic. If the user explicitly names a source or issuing authority, treat it as a constraint and prefer its matching endpoint over a different provider's similar result.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding

- For geographic fields that encode administrative hierarchies, follow the endpoint's documented hierarchy rather than treating city, province/state, metro area, and country as interchangeable. Use canonical codes where expected and provide the divisions needed to identify the requested place; use documented geographic lookup data to resolve names when available.
- When a schema splits one supplied identifier across component fields, parse it according to those field definitions and pass each component only to its corresponding field (for example, a username/local part and a domain). Do not repeat the entire compound identifier in each component field.

- When a request scopes a language or localized response to a country or market, use the supported locale that matches that market (for example, `en-US` for the US and `en-GB` for the UK), rather than a bare language code.
- Use each endpoint's exact parameter names and required/optional placement. Build the required-parameter list from fields the schema marks required, and put schema-optional fields only in the optional list—even when the user explicitly requests them. A requested value does not make an optional parameter required. Bind every constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful free text unless the schema requires a documented conversion. For categorical filters, codes, and selectors, map the request to the documented parameter meaning—not a similarly named field—and use the endpoint's canonical accepted token and casing from its enums or examples. Do not pass a descriptive word as a category value when it changes the field's meaning; preserve original wording when no conversion is documented. Interpret directional constraints according to field semantics.
- Include an optional parameter whenever it directly encodes a requested facet, even if the user phrases it in prose or the endpoint has a default. For an unfiltered, random, or full/standard result, use a documented “all” or unfiltered selector when needed to preserve that scope. Do not add options that are neither requested nor needed to express it, or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.

- In this answer format, encode scalar `value` fields as JSON strings, including numeric and boolean-like parameter values, unless the task's response contract explicitly requires native JSON numbers or booleans.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.

<!-- SLOW_UPDATE_START -->
When a task narrates a workflow or says what the user already searched for, treat its enumerated data needs as actionable unless it clearly asks only for acknowledgment; do not return an empty call list merely because the wording is past tense. For unavailable named endpoints, fulfill only the operations supported by relevant available tools—never substitute a tangential endpoint, such as past events for future events.

Resolve similar tools by matching both the input object and the requested result, not just the topic or provider. In particular, distinguish verifying an individual address’s deliverability from checking whether a domain is disposable; use a regular-dice endpoint for ordinary dice and a custom-dice endpoint only when nonstandard sides are requested; and select a forecast endpoint that supports the requested location and duration rather than an endpoint that only approximates the request.

Use the endpoint’s documented canonical tokens for enums and identifiers rather than guessing from natural-language labels or copying a full hostname. For example, map a marketplace hostname to its documented marketplace code, and use the documented token for an Ace or an unrestricted draw. Preserve meaningful search text unless the schema documents a conversion.

When serializing calls, put only schema-required fields in the required-parameters list; explicitly requested fields remain optional when the schema marks them optional. Bind each requested facet to its own supported field, including separate units, locale, pagination, and filters. Before answering, check that no requested facet was omitted, no optional field was promoted to required, and each value uses the response contract’s required representation—do not emit numeric or boolean-like values as native JSON unless that contract calls for them.
<!-- SLOW_UPDATE_END -->
