# Travel Planning Skill

## Cache-aware tool workflow
- At each turn, inspect cache summaries before searching. If a relevant result exists for the same domain, city, dates, and itinerary leg, retrieve it with its exact key using `get_results_from_cache`, then apply the matching `filter_*` tool with that result as `prior_results` and the newly supplied constraints. If no suitable cached result exists, run a new search. Apply new preferences to all relevant cached results or itinerary legs, but do not repeat valid searches or regenerate unaffected domains.
- When the user requests proximity, use the relevant hotel and attraction or restaurant results with `search_nearest`. Set `groupBy` to the domain the user wants prioritized.

## Tool results and constraints
- Assign every search, filter, sort, and nearest-results output to a variable, then immediately save it with `save_to_cache` under a unique, descriptive key before using it in later calls. Pass the assigned prior result to filters, use distinct variables and keys for separate domains or legs, and do not overwrite valid cache entries unintentionally.
- Pass only criteria supported by the conversation, map each value to the field that expresses it, and preserve the user's entity text. For example, put a state abbreviation in `state`, not `city`. Use the tool's accepted format for preferences such as time ranges; use an example format only when supported by the tool. Distinguish hotel stars (`stars`) from guest ratings (`rating`); for a supported “at least N stars” constraint, use the inclusive range from N to the field's maximum, and for an exact category use the same value as both bounds. Do not add unrelated preferences.

## Flight itinerary legs
Treat each explicitly described flight leg as a separate directional search with its own origin, destination, and departure date. For a round trip, search the outbound leg on the departure date and the return leg in the reverse direction on the return date; do not represent the return as `arrival_date` on the outbound search. Apply an explicitly requested flight class to both legs, and do not combine distinct legs by repeating keyword arguments in one call.

## Required-argument checks
Before calling a travel tool, check its required parameters against the full conversation and available context. If any required details are missing, use `seek_information` to ask only for those details; do not invent values, pass `None`, or run a partial search. For requests spanning multiple domains, bundle missing required details for all requested domains into the same information request. Do not ask for optional preferences.
