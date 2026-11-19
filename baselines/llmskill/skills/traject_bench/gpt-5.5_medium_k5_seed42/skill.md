### Work method

- First scan the tool list and build a small mental index for each endpoint: exact displayed name, main action, object it operates on, and what required inputs it needs.
- Then split the query into atomic requests. Treat each distinct action, endpoint-specific lookup, list retrieval, validation, search, or detail request as one candidate call.
- Narrative order words are usually not dependencies here. If the query says an action was performed as part of the desired workflow, include the corresponding call even if phrased in past tense.
- If the same action is requested for multiple independent values, make one call per value.
- If a request needs a result from another call that is not explicitly provided in the query, do not invent it. Only call tools whose required inputs can be filled from the query.

### Choosing tools

- Match by exact capability, not by broad topic. Compare the request’s verb and object with the provider/API name, description, and parameter requirements.
- Prefer the endpoint that directly performs the requested operation over a more general or adjacent endpoint.
- Distinguish carefully between:
  - searching/listing many items vs retrieving details for one known item;
  - reviews, prices, images, availability, categories, filters, reference lists, and help/documentation;
  - full validation/deliverability checks vs syntax-only checks vs disposable checks vs infrastructure-record checks;
  - address-level inputs vs domain-level inputs vs split local-part/domain inputs;
  - provider-specific searches vs the same type of search on another provider.
- Do not call setup, autocomplete, reference, or lookup endpoints unless the query explicitly asks for that information or the requested endpoint itself is the direct match.
- Ignore tools that merely sound related if their required input type or described output does not match the request.

### Filling required inputs

- For every selected call, locate each required input in the query before committing to the call.
- Preserve the query’s value as written whenever possible: identifiers, codes, domains, dates, timestamps, postal codes, search phrases, numbers, booleans, and marketplaces should not be translated, abbreviated, expanded, reformatted, or unit-converted.
- When a required input expects part of a compound value, split only when the tool description clearly requires it. For example, if an input must be a domain, use only the domain portion; if separate local and domain fields are required, split an address at the separator.
- If a tool requires a page, count, language, region, or similar value and the query does not provide one, do not guess unless the query explicitly states a default-like choice to use.
- If two values could fit one required input, choose the one tied most directly to that request, not a nearby value from another request.

### Optional inputs

- Add optional inputs only after the needed calls and required inputs are correct.
- Include optional inputs for explicit filters, limits, sorting, locale/region display choices, pagination, flags, postal/location refinements, or other modifiers stated in the query.
- Do not use optional inputs to compensate for missing required information.
- Since optional inputs are secondary, never let uncertainty about an optional modifier cause an extra tool call or a wrong required value.

### Final check

- Re-read the query and mark each atomic request as covered by exactly one call, except repeated values which need repeated calls.
- Re-read each selected tool block and verify every required input name is present with a value taken from the query.
- Check for extras: remove any call that exists only because it is related, preparatory, or potentially useful but not requested.
- Check for omissions: list requests with no required inputs still need calls when explicitly requested.
- Ensure repeated calls to the same endpoint differ only because the query supplied distinct values or distinct requested modifiers.
- Use the exact displayed tool names from the tool list when producing the final answer.
