# Tool Calling Skill

## Planning, tool selection, and validation
- Break the request into distinct operations and match each to an endpoint by its documented capability and parameter schema—not merely by provider or broad topic. Use the exact full tool identifier from the available-tools list. Do not substitute a tangential endpoint or call several similar alternatives for the same requested result.

- Before selecting calls, make a coverage ledger: list each requested result and its explicit constraints and facets (for example, geographic levels, time horizon, price or shipping filters, and requested metrics). Map each entry to the endpoint and schema field that provides it. Count an entry as covered only when the endpoint documents that exact result and scope; related or adjacent data do not suffice. If no single endpoint covers every entry, use separate appropriate calls. In the final audit, verify that every ledger entry is covered and every requested constraint is bound to a supported field.

- For overlapping endpoints, compare the exact input type and every requested output facet, not just the broad topic. If the user explicitly names a source or issuing authority, treat it as a constraint and prefer its matching endpoint over a different provider's similar result.
- For multiple requested checks, use an endpoint that covers them all only when its documented scope supports that. Otherwise, make separate calls. Batch multiple items only when the endpoint explicitly supports batching those resources with the applicable options; do not pack multiple calls into a scalar parameter.
- Before answering, check that every requested operation has an appropriate call and that no call serves an unrelated operation.

## Parameter binding

- For geographic fields that encode administrative hierarchies, follow the endpoint's documented hierarchy rather than treating city, province/state, metro area, and country as interchangeable. Use canonical codes where expected and provide the divisions needed to identify the requested place; use documented geographic lookup data to resolve names when available.

- For location-specific weather alerts, use a supported `point` filter for a named locality or area; use a state filter only when the requested scope is state-wide. Supply coordinates from the request or resolve them with a reliable geographic lookup rather than substituting the broader state or inventing coordinates.
- When a schema splits one supplied identifier across component fields, parse it according to those field definitions and pass each component only to its corresponding field (for example, a username/local part and a domain). Do not repeat the entire compound identifier in each component field.
- Use each endpoint's exact parameter names and required/optional placement. Build the required-parameter list from fields the schema marks required, and put schema-optional fields only in the optional list—even when the user explicitly requests them. A requested value does not make an optional parameter required. Bind every constraint only to the call it modifies; do not carry dates, limits, filters, or pagination across operations.
- Preserve user-provided identifiers and meaningful free text unless the schema requires a documented conversion. For categorical filters, codes, and selectors, map the request to the documented parameter meaning—not a similarly named field—and use the endpoint's canonical accepted token and casing from its enums or examples. Do not pass a descriptive word as a category value when it changes the field's meaning; preserve original wording when no conversion is documented. Interpret directional constraints according to field semantics.
- Include an optional parameter whenever it directly encodes a requested facet, even if the user phrases it in prose or the endpoint has a default. For an unfiltered, random, or full/standard result, use a documented “all” or unfiltered selector when needed to preserve that scope. Do not add options that are neither requested nor needed to express it, or invent values. For missing required values, use only a documented default or a value unambiguously derivable from the query.

- As a narrow exception, include an optional parameter when the endpoint explicitly recommends supplying it for successful or faster results and its value can be safely derived from context. For example, when a ski-snow endpoint recommends an explicit unit selector, use the documented customary unit for the named location if the user has not specified a preference; do not guess among unsupported values.

## Final response check
- Audit every call against its own schema and the decomposed request checklist. Ensure each call is relevant and has the exact full tool identifier and correctly formed parameter entries.

- In this answer format, encode scalar `value` fields as JSON strings, including numeric and boolean-like parameter values, unless the task's response contract explicitly requires native JSON numbers or booleans.
- Return a parseable JSON list inside the required tags, with exactly the required keys (`tool name`, `required parameters`, and `optional parameters`) in each call object. Include no commentary or reasoning inside the JSON list.

## Call dependencies
- Treat calls as independent unless the request or API requires a dependency.

<!-- SLOW_UPDATE_START -->
Make one checklist row for every requested operation, not merely every topic or product. Treat narrated workflows as requested operations unless the user clearly asks only for acknowledgment; when both a broad search and a specific item lookup are requested, include both. Before finalizing, match every row to a documented endpoint and verify that the call covers that row’s exact result and scope.

When tools have similar names, compare their full provider-qualified identifiers, documented input type, and output—not just the endpoint’s short name. Distinguish endpoints such as “latest” versus “relevant,” singular versus plural, and similarly named tools from different providers. Do not add several generic alternatives for one check or let a related endpoint stand in for a requested operation.

For comprehensive email or business-screening requests, separate address-level checks from domain-level checks. Cover each distinct supported facet the user requests or clearly implies—such as address validation, disposable-domain status, role-account status, MX records, or SMTP—and pass a full address or domain according to that endpoint’s schema. Do not treat a generic email validator as covering every domain check, and do not call every available validator just because the user says “comprehensive.” For fake-address generation, use the documented generator endpoint, such as `Email API: Get Fake Email Address`, rather than a similarly named tool from another provider.

Treat codes, enums, source identifiers, and marketplace selectors as API tokens: use the documented spelling and casing, not the user’s display text. Check that publisher fields use the documented identifier format; map a marketplace host such as `amazon.de` to the documented country token when the schema expects one. For broad location searches, use the documented all-results selector rather than a narrower type. Map “since” dates to the start-date field and reserve end-date fields for “until” or “before” constraints. Keep every filter in the schema’s required or optional list as documented, even when the user explicitly requests that filter.

In the final audit, check each call’s exact tool identifier, input granularity, canonical values, parameter placement, and correspondence to a checklist row. Remove unrelated or redundant calls, and add any missing operation before returning the required JSON.
<!-- SLOW_UPDATE_END -->
