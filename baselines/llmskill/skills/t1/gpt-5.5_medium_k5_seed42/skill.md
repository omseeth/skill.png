### Planning procedure

- Read the whole conversation, but make the plan for the latest user turn. Treat earlier user facts as still active unless the user changed them.
- First decide the turn type:
  - New domain request with all mandatory inputs known: run the relevant search tool.
  - New constraint on an existing result: retrieve the most relevant cached result and use the matching filter tool.
  - Request for cheapest, highest rated, shortest, earliest, etc.: retrieve or create the relevant result, then use `sort_results`.
  - Request for proximity between two result sets: retrieve or create the needed sets, then use `search_nearest`.
  - Missing mandatory input for a requested action: use `seek_information` for only those missing mandatory inputs. If another requested domain is already actionable, still plan it.
  - If the cache already contains exactly the result needed and the last user turn adds no new actionable constraint, do not create or save anything new.
- Only operate on domains the user has actually requested in the conversation.

### Mandatory inputs to check

- `search_flights`: needs `departure_date` and at least one start location field: `start_airport_city`, `start_airport`, or `start_airport_code`.
- `search_hotels`: needs `city`, `checkin_date`, and `checkout_date`.
- `search_restaurants`: needs `city`.
- `search_attractions`: has no non-optional search input; use any explicit city/state/name/type/neighborhood provided.
- Filter tools need `prior_results`; get them from cache unless produced earlier in the same plan.
- `sort_results` needs `results` and `column`.
- `search_nearest` needs `groupBy` and at least two relevant result sets among hotels, restaurants, attractions.
- `adjust_date` needs `date` and `delta_days`; use it when the user gives a stay length or a relative date.

### Cache use

- Use the cache summary, not assumptions about hidden results, to choose the best prior result.
- Prefer the most specific cached result that matches the current domain and itinerary. If multiple keys fit, choose the one whose summary already includes the most applicable constraints.
- When the last user adds constraints to a cached search, filter the cached result instead of repeating the original search.
- Never reference variables from prior turns; retrieve cached results with `get_results_from_cache`.
- Save every new search, filter, sort, or nearest result exactly once under a key not already present. Use simple unique names that reflect the domain/action.
- Do not save outputs from `get_results_from_cache`, `adjust_date`, or `seek_information`.

### Parameter extraction

- Include every explicit user constraint that maps to a tool parameter; omit optional parameters that were not explicitly provided.
- Preserve user-provided entity strings exactly for cities, neighborhoods, names, airlines, airports, cuisines, attraction types, and hotel/restaurant names. Do not canonicalize spelling, casing, abbreviations, or synonyms.
- Use lists where the tool expects lists, even for one value.
- Budgets are maximums: pass numeric values to `budget` only when the user states a price limit.
- Ratings:
  - “at least”, “minimum”, “or higher”, or a bare desired rating usually means `[value, 5]`.
  - “between” means `[low, high]`.
  - Preserve integer vs decimal when clear, especially for restaurant ratings.
- Hotel stars:
  - “at least N stars” means `[N, 5]`.
  - “N-star hotel” or “exactly N stars” means `[N, N]` unless the wording clearly means minimum.
- Times:
  - Convert AM/PM times to `HH:MM:SS`.
  - “by” or “before” an arrival/departure time means `["00:00:00", time]`.
  - “after” means `[time, "23:59:59"]`.
  - “between X and Y” means `[X, Y]`.
- Flights:
  - Direct/nonstop means `num_layovers=[0]`.
  - One stop means `num_layovers=[1]`; up to one stop means `[0, 1]`; two stops means `[2]`.
  - Airline preferences go in `airline` as a list.
  - Cabin preferences go in `flight_class` using the user’s class wording.
  - Layover city preferences go in `layover_1_city` / `layover_2_city`; only include layover durations when explicitly limited.
- Hotels:
  - Map amenities directly to booleans only when stated. Examples: breakfast, gym, pool, wifi, shuttle, pet friendly, spa, room service, beach access, business center, laundry, valet, balcony, rooftop bar, kitchen, kids club, meeting rooms, EV charging, hot tub, sauna, parking, wheelchair access, skiing lodging, ocean/city view.
  - “No smoking” / “non-smoking” means `smoking_allowed=False`; “smoking allowed” means `True`.
  - Use `num_people` and `num_rooms` only when explicitly stated.
- Restaurants:
  - Cuisine, neighborhood, dietary restrictions, allergy options, and price/rating preferences map directly to restaurant parameters.
  - Vegan/vegetarian/gluten-free/kosher/halal/allergy requirements are boolean `True` when requested.
- Attractions:
  - Use `type`, `attraction_name`, `city`, `state`, and `neighborhood` only from explicit user information.

### Dates and itinerary logic

- For hotel stays given as a number of nights, compute checkout with `adjust_date(date=<checkin>, delta_days=<nights>)`, then use that variable in `checkout_date`.
- For a hotel date range, the first date is check-in and the second is checkout.
- For a one-way flight, use the stated departure date.
- For a round trip or wording that clearly includes leaving and returning, create two flight searches: outbound on the first travel date and return on the return date with reversed endpoints.
- For multi-city flight legs with date ranges per leg, use the first date of each leg’s range as that leg’s departure date unless the user explicitly asks to search all dates in the range.
- If the user gives alternative possible dates, pass all alternatives in the date list.
- If a date is derived from another date, use `adjust_date`; do not manually calculate when a tool call would make the dependency clear and executable.

### Filtering vs new searching

- Use a filter when the current request narrows, re-ranks after narrowing, or adds preferences to an already searched result set.
- Use a new search when the core entity changes: different city/route/domain, a separate leg, or no suitable cached result exists.
- If a new filter should apply to multiple cached results in the same domain because the user’s preference applies to all of them, filter each relevant cached result and save each output.

### Sorting and nearest

- Sort columns:
  - Flights: cheapest by the relevant class price, fewest stops by `num_layovers`, shortest by `duration_minutes`, earliest/latest by date or time columns.
  - Hotels: cheapest by `price_per_night`, best by `rating_hotels` descending, highest stars by `stars` descending.
  - Restaurants: cheapest by `price_per_person`, best by `rating_restaurants` descending.
- Ascending is for cheapest, fewest, shortest, earliest, lowest; descending is for highest, best, latest, most expensive.
- For nearest requests, `groupBy` is the result type the user wants returned/grouped, e.g. hotels near attractions uses `groupBy="hotel"`.

### Final self-check

- Confirm no required mandatory value is missing for every non-`seek_information` tool call.
- Confirm every parameter value came from the conversation or from an executed tool result.
- Confirm no optional defaults were added just because the tool has defaults.
- Confirm every result-producing call is saved under a new key and no extra saves are present.
- Confirm code would run: keyword arguments for tools except `seek_information`, valid Python booleans, no imports, no undefined prior-turn variables, and no attempts to inspect hidden result contents.
