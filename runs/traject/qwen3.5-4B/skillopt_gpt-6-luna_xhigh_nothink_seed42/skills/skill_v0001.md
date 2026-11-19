# Tool Calling Skill

## Plan tool calls
- Turn the query into a checklist of distinct requested outcomes, including each subject, action, and explicit constraint. Keep modifiers attached to the result they qualify. Match each outcome to an available endpoint by its actual operation and schema, not just a shared keyword; distinguish similar operations and prefer an explicitly named endpoint when available.
- Build the smallest complete call set: cover every requested outcome, and make every call serve one of them. Use one endpoint for multiple outcomes only when it genuinely covers them; make separate calls for distinct inputs unless the schema supports batching. If a requested endpoint is unavailable, use a listed alternative only if it genuinely fulfills the request. Do not invent tools, add unrelated calls, or include redundant alternatives or validators.

- When calls are independent, issue them together rather than treating narrative order as a dependency.

## Ground arguments in the request and schema
- Preserve requested search text, identifiers, and constraints; normalize only as the endpoint requires, without dropping meaningful words or substituting one value or geographic level for another. Map explicit filters, counts, pages, sorting, locale, category, and location to the corresponding schema fields, using accepted representations for enums and locales.

- Translate stated market constraints into API-native values according to the tool schema.
- Represent parameters in the schema's required form and types. Put each in the declared required or optional bucket, and do not duplicate parameters across buckets. Include optional settings only when requested or clearly implied; use documented recency or freshness controls for requests such as “recent” or “latest,” and a modest supported result count when wording such as “some” calls for one. Do not invent dates, defaults, limits, IDs, credentials, filler values, or other unsupported constraints. If a required value is missing, do not fabricate it.

## Validate before responding
- Check that every requested outcome is covered, each call uses the exact listed tool name, and each call object follows the active answer format exactly. Use the required keys, parameter buckets, types, and wrappers; include empty arrays when required. Do not merge calls, duplicate keys, or add explanatory text outside the required format.
