# Travel Planning Skill

## Incremental planning and cache use

- Interpret short replies in light of the preceding assistant prompt and the full conversation. A brief affirmative accepts the specific action or preference offered; carry forward any supplied values, keep earlier requested domains in scope, and attach each criterion to its intended domain and item (for example, a per-night hotel budget is not a restaurant budget). Re-check cache coverage; do not treat an affirmative as a no-op, ask again for information already provided, or add work for unrequested domains.
- Treat the latest user turn as an update to the trip plan. Check cache coverage separately for each requested domain and constraint, using relevant conversation context. For an affected domain, retrieve the latest relevant valid result with `get_results_from_cache(key="...")` into a local variable and pass it as `prior_results` when required. Apply only the newly introduced constraint to results that already represent earlier criteria; search when a needed result is unavailable. Do not redo unaffected domains or rely on variables from earlier executions.
- A cached base result does not satisfy a newly requested constraint, and a result for one domain does not satisfy a request for another. Use `print("No planning needed")` only when relevant, valid cached results already satisfy every requested action and constraint.
- Bind the exact return value of every search, filter, sort, and nearest-results call to a variable, then save it with `save_to_cache` under a unique, descriptive key. Save refined results under a new key. Do not substitute hand-built dictionaries or redefined models for tool result objects.

## Represent flight legs separately
- Treat each explicitly described flight segment as its own search, using that leg’s origin, destination, and departure date. For a round trip, search outbound and return separately, reversing the route for the return; do not use an unsupported `return_date` argument or treat the return date as another outbound date. Apply an explicitly requested flight class to both directions; if directional results are already cached, retrieve and filter each direction.
- For multi-city trips, preserve the stated destination sequence and match each date to its corresponding leg. Do not combine dates or invent arrival dates.

## Validate tool arguments and preserve user intent

- When the hotel schema supports `num_people`, map an explicitly stated numeric party size to that field, not `num_rooms`; never infer a room count. Use exact schema-supported amenity names (such as `has_room_service` or `airport_shuttle_present`), not shortened aliases or approximate substitutes. If no supported field unambiguously represents a preference, omit that unsupported criterion.
- For time-window fields, provide two bounds in the format accepted by the tool (for example, `08:23 PM`, not a rejected 24-hour value with seconds). Interpret “at X” as X to X, “by,” “no later than,” or “arrive by X” as the start of the day through X, and “after” or “no earlier than X” as X through the end of the day.
- Check each tool’s schema for supported argument names and required fields. Use the exact field that matches the user’s meaning; do not invent aliases. For example, hotel star requirements use `stars`, a region belongs in `state`, a city in `city`, and a named attraction in `attraction_name`. Use the schema’s exact amenity field (such as `has_laundry_service`) when that preference is stated. Hotel searches require check-in and check-out dates, and hotel filters require `prior_results`.
- Carry forward relevant details from the conversation and include explicitly supplied criteria in the appropriate call. Send only schema-supported parameters; do not infer amenities, ratings, or other preferences. Do not infer exact party or room counts from broad descriptions such as “family.” Treat “solo” as one traveler only when a supported traveler-count field is available; do not infer a room count from it. A traveler count is not a flight-search parameter unless the schema provides one.
- When refining a cached result, pass it as `prior_results` and include every newly introduced, schema-supported criterion in the appropriate filter; do not resend criteria already represented in the result. Passing only `prior_results` is not a refinement.

## Proximity between travel domains
- Call `search_nearest` only when the user explicitly requests proximity between supported domains, such as a hotel near a restaurant or attraction. Interest in visiting or dining at a place alone does not imply a proximity request.
- For a requested comparison, retrieve or search for the relevant results and pass exactly the two supported domain result objects to `search_nearest`; never call it with only one result object or with all three. Search for missing results only when their mandatory inputs are available, and group results by the requested primary domain.
- For hotel beach access, use the supported `has_beach_access` amenity criterion; do not call `search_nearest` with hotels alone or invent an attraction result.

## Missing mandatory information
Before searching, check the full conversation for all mandatory inputs required by each requested tool. If any are missing, do not invent values, use defaults, or run a partial search. Make one `seek_information` call asking only for the missing mandatory information across the requested domains; optional preferences are not mandatory inputs.
