# Tool Calling Skill

## Match requested results to endpoints
- Identify each distinct requested result or check and choose the catalog endpoint whose documented operation and inputs fit. A related topic or provider alone is not enough; distinguish different outputs such as validation versus disposable status, or product details versus similar items.
- When the user names a tool or service, use its matching endpoint. Copy the exact tool name from the catalog, including provider and endpoint.
- Call only for requested outcomes. Do not add overlapping endpoints just in case; make another call only when it provides a separately requested result the first call does not cover.

## Follow the endpoint schema
- Map every explicit constraint—such as dates, locale, counts, units, pages, and filters—to the corresponding documented parameter. Do not omit a requested setting just because the endpoint has a default.
- Use the exact parameter names, required or optional classifications, types, and accepted value formats documented for the endpoint. Do not substitute guessed synonyms or assume a display name can be converted into an API value by an intuitive spelling change. Use values grounded in the request or documentation, and preserve free-text values unless normalization is specified.
- For a requested number of items, prefer an endpoint that supports that count and set its documented count parameter rather than repeating a single-item call. Omit blank or unrelated optional parameters.

## Validate calls and answer structure
- Use the active answer wrapper and schema, with one call object per endpoint invocation. Do not combine multiple calls or tool names in one object.
- Before submitting, check that every planned call is represented, the final JSON parses, and each call has the required interface structure with exactly one entry for each parameter collection. Remove duplicate keys.
