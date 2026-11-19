# Travel Planning Skill

## Incremental planning and cache use

- Before emitting code, audit the planned calls against the latest turn and full conversation: for each new request or accepted preference, identify its intended item, domain, relation, and schema-supported argument, then verify that a planned call includes it. Do not reprocess unchanged domains. Use only a listed, compatible cache result for refinement; if none exists and mandatory inputs are available, search instead of filtering. After the audit, use `print("No planning needed")` only if no requested action or constraint remains unsatisfied by valid cached results.
- Treat the latest user turn as an update to the trip plan. Check cache coverage separately for each requested domain and constraint, using relevant conversation context. For an affected domain, retrieve the latest relevant valid result with `get_results_from_cache(key="...")` into a local variable and pass it as `prior_results` when required. Apply only the newly introduced constraint to results that already represent earlier criteria; search when a needed result is unavailable. Do not redo unaffected domains or rely on variables from earlier executions.
- A cached base result does not satisfy a newly requested constraint, and a result for one domain does not satisfy a request for another. Use `print("No planning needed")` only when relevant, valid cached results already satisfy every requested action and constraint.
- Bind the exact return value of every search, filter, sort, and nearest-results call to a variable, then save it with `save_to_cache` under a unique, descriptive key. Save refined results under a new key. Do not substitute hand-built dictionaries or redefined models for tool result objects.

## Represent flight legs separately
- Treat each explicitly described flight segment as its own search, using that leg’s origin, destination, and departure date. For a round trip, search outbound and return separately, reversing the route for the return; do not use an unsupported `return_date` argument or treat the return date as another outbound date. Apply an explicitly requested flight class to both directions; if directional results are already cached, retrieve and filter each direction.
- For multi-city trips, preserve the stated destination sequence and match each date to its corresponding leg. Do not combine dates or invent arrival dates.

## Validate tool arguments and preserve user intent
- Check each tool’s schema for supported argument names and required fields. Use the exact field that matches the user’s meaning; do not invent aliases. For example, hotel star requirements use `stars`, a region belongs in `state`, a city in `city`, and a named attraction in `attraction_name`. Use the schema’s exact amenity field (such as `has_laundry_service`) when that preference is stated. Hotel searches require check-in and check-out dates, and hotel filters require `prior_results`.
- Carry forward relevant details from the conversation and include explicitly supplied criteria in the appropriate call. Send only schema-supported parameters; do not infer amenities, ratings, or other preferences. Do not infer exact party or room counts from broad descriptions such as “family.” Treat “solo” as one traveler only when a supported traveler-count field is available; do not infer a room count from it. A traveler count is not a flight-search parameter unless the schema provides one.
- When refining a cached result, do not resend criteria already represented in that result; apply the newly introduced preference with the appropriate filter.

## Proximity between travel domains
- Call `search_nearest` only when the user explicitly requests proximity between supported domains. Before calling it, verify that each cached result covers the specific requested target, not merely the same domain or city. For example, a generic city attraction search does not cover a separately named attraction; search for the named attraction using `attraction_name` unless the cache summary shows that exact target is already covered.
- Pass the relevant result objects for the two domains being compared. Set `groupBy` to the domain whose candidates the user wants ranked or recommended (the focal domain), not the reference domain: hotels near an attraction use `groupBy="hotel"`, while restaurants near a hotel use `groupBy="restaurant"`.
- Interest in visiting or dining at a place alone does not imply a proximity request or require a nearest-results call.

## Missing mandatory information
Before searching, match each required tool field to an explicit value in the full conversation; related context does not automatically satisfy that field. For example, a reunion date is not necessarily a flight departure date, and restaurant cuisine or budget does not supply the required restaurant city. Do not infer missing mandatory values or use defaults. If any are missing, make one `seek_information` call asking only for the missing mandatory information across the requested domains; optional preferences are not mandatory inputs.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
