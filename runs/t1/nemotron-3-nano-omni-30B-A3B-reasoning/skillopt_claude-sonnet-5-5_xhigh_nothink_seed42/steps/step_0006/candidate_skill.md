# Travel Planning Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Rule 6: Never fabricate mandatory parameters (dates especially)
- Before any search_* call, check that every mandatory argument was EXPLICITLY stated by the user in the conversation. Never invent, guess, or default values, even for 'last-minute', 'this weekend', or 'business trip' requests with no concrete date.
- Dates in tool usage examples (e.g. '2025-08-03', '2025-08-11') are illustrations only. Never copy them into generated code.
- Mandatory args that are commonly missing:
  - search_flights: departure_date, plus an origin (start_airport_city / start_airport / start_airport_code).
  - search_hotels: city, checkin_date AND checkout_date.
  - search_restaurants: city.

- A restaurant request that gives only cuisine, budget, rating or dietary needs ('I'm looking for German food', 'we want to spend around 70') and no city anywhere in the conversation is missing the city. Output only `seek_information("We need the city for the restaurant.")`. Never invent a city or borrow one from the tool examples (Dallas, Boston, Phoenix, ...). Do not call search_restaurants without a user-stated city.
- Check mandatory args PER DOMAIN. If every requested domain is missing something (or the only requested domain is incomplete), output ONLY a seek_information(...) call naming what is missing, with no search call and no placeholders. If SEVERAL domains were requested and some already have all their mandatory args (e.g. 'hotel in Colorado Springs plus restaurant recommendations in the city' gives the restaurant city but no hotel dates), first run `x = search_*(...)` + `save_to_cache(key="<domain>", value=x)` for each complete domain. Then add ONE seek_information(...) at the end naming only the incomplete domain's missing items. Example: `restaurants = search_restaurants(city="Colorado Springs")`, `save_to_cache(key="restaurants", value=restaurants)`, `seek_information("We need the check-in and check-out dates for the hotel.")`. Never call search_hotels / search_flights with guessed values to fill the gap.
- Do not derive a date with adjust_date unless the user gave a reference date and an offset.
- Optional details (budget, stars, rating, amenities, neighborhood, num_people, num_rooms, cuisine, flight class) never substitute for missing mandatory args and are never a reason to call seek_information. A user who has answered several follow-ups about budget, stars, or amenities may still not have given dates, so re-check each time.
- A round-trip request still needs the departure date. Ask for it; do not assume a return date.

- A hotel needs its own explicit checkout date. A flight departure date or arrival date supplies at most the check-in date. It never supplies a checkout date, and a one-night stay is never a default. If a hotel is requested and only one date is known (e.g. 'fly on May 13, 2025, hotel near Downtown'), do NOT call search_hotels with a guessed checkout and do NOT infer num_people from context. Output `seek_information("We need the check-out date for the hotel.")`, naming only what is missing. If flights for the stated date are already cached, leave them as they are; do not re-filter them with a budget or other constraint their CACHE summary already lists.

- Context is not a date. A flight request with origin and destination but no date is still missing departure_date, even if the user also gave 'one-way', a party size ('2 travelers'), a purpose ('wedding', 'family reunion', 'business trip'), a timing worry ('make sure we arrive on time'), or a home-airport habit ('I normally fly from LEX'). Output only `seek_information("We need the departure date for the flight.")`. Do not search, and do not ask about class, airline, time, return date, or passenger count.

- An event date is not a travel date. 'The reunion is on May 26, 2025', 'the wedding is June 3' or 'the conference starts May 5' do not give the flight departure_date or the hotel check-in/check-out dates. Unless the user says they fly or check in on that day, still output only the seek_information for the missing departure date (or hotel dates).

## Rule 7: Writing seek_information, including multi-domain requests
- Mention only MISSING MANDATORY parameters. Never ask for optional ones.
- When the user asks for several domains (flight + hotel, hotel + attraction, etc.), check the mandatory args of each requested domain separately. If any domain is missing something, list ALL missing items across domains in ONE seek_information call. Do not run a partial search with made-up values for another domain.
  Example: seek_information("We need the departure date for the flight, and the check-in and check-out dates for the hotel.")
- Reuse values the user has actually stated and ask only for what is still missing.
- Keep it to one short sentence, e.g. seek_information("We need to ask for the check-in and check-out dates") or seek_information("We need the departure date for the flight").

## Rule 8: Act on the latest turn and the cache
- Decide what to call from the LAST user turn and the cache summary, not from the whole conversation history.
- If the last turn adds no new constraint (e.g. 'Not really, just something fun', 'Okay, that sounds good', 'No preference') and the cache already holds results for the domain under discussion, do NOT re-run search_* with the old criteria and do NOT print("No planning needed"). Retrieve the stored results with `x = get_results_from_cache(key="<key>")`, using the most refined key that matches the user's request (e.g. key="attractions"). No save is needed for a pure retrieval.
- If a domain already has a cache entry (e.g. 'flights' for the same route and dates) and the last turn concerns another domain (e.g. hotel amenities), do not re-run the cached domain's search. Retrieve it only if needed.
- Call search_* / filter_* again only when the user introduces a new or changed constraint. A new constraint on a cached domain is handled by filtering (Rule 2).
- If the last turn only gives a preference for a domain that still lacks mandatory values (e.g. 'a hotel with a gym and wifi' but no dates), call seek_information for the missing values (Rules 6 and 7) instead of searching.

## Rule 2: Refinement turns load from the cache and filter (never re-search, never use undefined variables)
- If the CACHE already holds results for a domain and the latest user turn adds a preference (amenity, stars, budget, flight class, airline, time of day, party size, neighborhood, cuisine, rating, etc.), do NOT call search_* again. Load the cached result, filter it, and save under a new key:
  `hotels = get_results_from_cache(key="hotels")`
  `hotels2 = filter_hotels(prior_results=hotels, pool_present=True, breakfast_included=True)`
  `save_to_cache(key="hotels2", value=hotels2)`
- Never reference a variable that was not defined in the current code block (e.g. `hotels` from an earlier turn). Always load first with `x = get_results_from_cache(key="...")`, then use `x`. The key must be copied exactly from the CACHE summary.
- Chain from the most refined and most recent key for that domain (the latest `filtered_*` or `nearest_*` key beats the raw search key). Never filter a stale, broader key.
- Criteria to pass to filter_*: the constraints introduced in the LAST user turn, plus any earlier-stated constraint that the CACHE summary text ("Additionally filtering for ..." / search description) does not show as applied. Do NOT repeat filters or search parameters the summary already lists. Never call filter_* with only prior_results.
- Map each stated preference only to its matching parameter. For example, 'kid-friendly pool' means pool_present=True. Do not add extras such as has_kids_club unless the user states them.
- Handle every domain and leg affected by the last turn. If the user answers with both an airline and hotel amenities, filter both. If departure_flights and return_flights are both cached, apply a shared flight preference (class, time of day, airline) to EACH leg with its own load, filter and save.
- If the last turn introduces a domain that has no cache key (e.g. restaurants after hotels, or attractions after flights), run search_* for that domain only, using the city already established in the conversation. Do not redo filters for domains that are already cached.

## Rule 14: Straight-line first search once mandatory args become complete (supplements Rule 11)
- When every mandatory arg for a domain is now available (values may come from earlier turns, e.g. city in turn 1 and dates in the last turn) and the CACHE has no key for that domain, emit just `x = search_*(...)` then `save_to_cache(key="<domain>", value=x)`, using a simple domain key such as "hotels", "flights", "restaurants" or "attractions".
- Do NOT probe the cache with a key you invented (e.g. "hotels_memphis_dec25"). Only keys that literally appear in the CACHE summary may be loaded. Do not add comment-only 'checks', and when all mandatory args are known do not add print() or seek_information fallbacks.
- Include optional details the user explicitly stated (num_rooms, num_people, budget, amenities, cuisine, ...) as keyword args in the search call, and nothing the user did not say.
- Search only the domain whose mandatory args just became complete. Leave other domains' existing cache entries untouched (e.g. restaurants already cached, now the hotel dates arrive: call search_hotels only).

## Rule 13: Refinement precision and code hygiene (supplements Rule 2)
- A refinement on a cached domain is exactly three straight-line statements: `x = get_results_from_cache(key="<exact key from CACHE>")`, `y = filter_<domain>(prior_results=x, <only the new constraints>)`, `save_to_cache(key="<new unique key, e.g. filtered_<domain>>", value=y)`. A neighborhood, budget, cuisine, rating or amenity answer on a domain that has a cache key is always a filter, never a fresh search_* with the old criteria re-listed.
- Load only keys of the domain being refined, starting from its most refined key (e.g. filtered_hotels, not hotels). Do not load or return unrelated keys (e.g. flights when only a hotel amenity was added). Do not repeat anything the CACHE summary already lists (city, stars, neighborhood, cuisine, amenities, traveler count), and do not add city to a filter.
- Flight legs can have any cache key names (departure_flights/return_flights, first_flights/second_flights, flights). When the user gives a class, time or airline preference, load EACH flight key from the CACHE, filter it, and save each under a new key. Never re-run search_flights. 'Flexible, but preferably in the afternoon' is still a time preference: `departure_time=["12:00 PM", "5:00 PM"]` on each leg.
- Only refine domains the last turn actually talks about. Never invent filter values (e.g. a made-up cuisine list or budget) for another domain.
- Never call sort_results unless the user explicitly asks to sort or rank (cheapest, highest rated, earliest). Never invent a sort column.
- Output only short, direct tool calls inside <CODE></CODE>. NEVER write `import`, `from ... import`, `class`, `def`, or any re-definition of tools or models, because the tools are pre-defined and this causes NameError. Never wrap cache reads in `if x is None` branches, never use if/else or loops, and never nest get_results_from_cache inside the filter call.
- Do not print results, loop over `.search_results`, or access result fields such as hotel['name'] or ['rating']. The result schema is unknown and key errors break the run.
- Do not write long reasoning in code comments; use at most a one-line comment per step. A response that ends mid-comment without finishing the calls is a failure.
- Use parameter names EXACTLY as listed in TOOL CONFIG. Hotel amenities other than gym_present, pool_present, breakfast_included, smoking_allowed, air_conditioning_present, heating_present, free_wifi_included, airport_shuttle_present and is_pet_friendly use the `has_` prefix: has_business_center, has_spa_services, has_rooftop_bar, has_free_parking, has_electric_vehicle_charging, etc. Writing `business_center=True` raises a validation error. Check each kwarg against the tool's arg list before writing it.

## Rule 10: Route the LAST turn's constraints to the right domain; a 'yes' to a proposal is a new constraint

- Loose numeric budgets are still budgets. 'around 500 per night', 'about 110', 'we're looking to spend around 110' -> `budget=500` / `budget=110` (plain integer, no rounding or range) on the domain being discussed: hotel budget is per night, restaurant budget is per person, flight budget is per ticket. Combine it with any other new constraint in the same turn in ONE filter_* call (e.g. `filter_hotels(prior_results=x, budget=500, has_laundry_service=True)`). If that domain has no cache entry and its mandatory args are known, pass budget in the first search_* call. If a mandatory arg such as the restaurant city is still missing, only seek_information for it.
- Before writing code, list each new constraint in the last user turn and assign it to a domain: cuisine, dietary needs, restaurant rating and restaurant budget -> restaurants; amenities, stars, hotel budget, rooms and neighborhood -> hotels; attraction names or types -> attractions; airline, class, times -> flights.
- For each affected domain, load THAT domain's most refined cache key and call that domain's filter_* tool. Never load or filter another domain's key, even if that domain was filtered last turn, and never add another domain's params (e.g. breakfast_included) as filler. Example: `r = get_results_from_cache(key="restaurants")`, `r2 = filter_restaurants(prior_results=r, cuisine=["Indonesian"], rating=[4.5, 5])`, `save_to_cache(key="filtered_restaurants", value=r2)`.
- Read the assistant's LAST message. A short affirmative ('Yes', 'That sounds wonderful', 'Yes, we'd like Downtown Memphis') that accepts a concrete proposal (a neighborhood, an attraction type, a cuisine, a flight class, an amenity) is a NEW constraint. This overrides the retrieval-only bullet of Rule 8. Never just retrieve from the cache and never print("No planning needed") in that case.
  - Neighborhood accepted: `x = get_results_from_cache(key=<latest hotels key>)`, `x2 = filter_hotels(prior_results=x, neighborhood=["<user's wording>"])`, then save under a new key.
  - Attraction type accepted (e.g. 'Scenic', 'Cultural') and no cached attractions entry matches it: `attractions = search_attractions(city=<established city>, type=["Scenic"])` then `save_to_cache(key="attractions", value=attractions)`. If a matching cache entry exists, filter it instead.
  - Only a refusal or 'no preference' ('Not really', 'Anything is fine') with an existing cache entry means pure retrieval.
- If the user names a specific attraction (and/or its type) that is not yet cached, call `search_attractions(city=<established city>, attraction_name=["<exact name>"], type=["<type if stated>"])` and save it under 'attractions'. Do not touch the hotel cache for that turn. If a named attraction arrives in the same turn as hotel amenities, do BOTH: filter hotels and search attractions.
- Pass only values the user literally stated in the last turn (plus unapplied earlier constraints per Rule 2). NEVER pass True/False for features nobody mentioned and never enumerate every amenity parameter. 'Wifi, a gym, and a rooftop bar' -> free_wifi_included=True, gym_present=True, has_rooftop_bar=True and nothing else. 'Free parking' -> has_free_parking=True only. 'Kid-friendly pool' -> pool_present=True only.
- Do not convert vague style words ('formal', 'fun', 'romantic', 'nice', 'budget-friendly' with no number) into rating, stars, budget or amenity params. 'Formal with a good Italian menu' -> cuisine=["Italian"] only.
- Rating phrasing: 'at least X' -> rating=[X, 5]; a plain 'X rating' -> rating=[X, X]. A stated cuisine -> cuisine=["<as stated>"]. When the user answers an assistant question that named a specific amenity, use the single param matching that wording ('fitness classes' -> has_fitness_class=True, 'gym' -> gym_present=True), not several near-synonyms.

## Rule 9: Round-trip flights (+ hotel) are separate search calls
- search_flights has NO return_date parameter. Never pass it. A round trip or date pair ('May 15 to May 19', 'leaving X, coming back Y') is TWO calls, never both dates in one departure_date list:
  `departure_flights = search_flights(start_airport_city="A", end_airport_city="B", departure_date=["May 15, 2025"])`
  `save_to_cache(key="departure_flights", value=departure_flights)`
  `return_flights = search_flights(start_airport_city="B", end_airport_city="A", departure_date=["May 19, 2025"])`
  `save_to_cache(key="return_flights", value=return_flights)`
  A multi-city trip needs one search per leg. If the user offers several candidate dates for ONE leg, put them all in that leg's departure_date list.
- Copy any shared preference (flight_class, airline, etc.) to BOTH legs. Do not add unmentioned optional params. If the user names an airport code (e.g. 'I normally fly from SJC'), add start_airport="SJC" on the outbound leg and end_airport="SJC" on the return leg.
- If a hotel is also requested anywhere in the conversation (even just 'and a hotel stay for a special occasion') and the dates are known, run `search_hotels(city=<destination city>, checkin_date=[<outbound date>], checkout_date=[<return date>])` in the same code block and save it under "hotels". Add num_people when party size is stated. Do not drop the hotel because the latest turn only mentions flight details (dates, class).
- Use date strings exactly as the user wrote them (e.g. "May 15, 2025").

## Quick decision order for every turn
1. List the domains the user has requested anywhere in the conversation (flight, hotel, restaurant, attraction). Ignore domains never mentioned.
2. For each domain, check the CACHE: a key exists -> refine with filter_* (Rules 2, 13, 17, 18) or retrieve only if the last turn adds nothing new (Rule 8). No key -> go to step 3.
3. Check that domain's mandatory args, using values from ANY earlier turn. All present -> `x = search_*(...)` plus `save_to_cache` (Rules 14, 15). Anything missing -> name only the missing mandatory items in ONE combined seek_information at the end of the block (Rules 6, 7). A domain whose args are complete is still searched in the same block; never search a domain with guessed values.
4. Pass only the constraints the user literally stated. Never ask about optional details, and never seek information for a domain the user has not requested.

5. A requested domain with NO key in the CACHE is a search, even when other domains' keys exist and the last turn only supplied dates or amenities. Example: CACHE holds only 'attractions'; the user asked for a hotel in Fresno for 3 people, budget 65, 4 stars; the last turn gives the dates. Output `hotels = search_hotels(city="Fresno", checkin_date=["August 17, 2025"], checkout_date=["August 20, 2025"], num_people=3, budget=65, stars=[4, 4])` + `save_to_cache(key="hotels", value=hotels)`. Never write `get_results_from_cache(key="hotels")` when 'hotels' is not in the CACHE. Gather every hotel constraint stated anywhere in the conversation (party size, budget, stars, amenities) into this first search call. If flights are already cached and the last turn only supplies the missing hotel checkout date (check-in = the known flight date), call search_hotels once and do NOT re-run search_flights or call filter_hotels.
6. City carry-over: the city for restaurants, attractions or hotels is the destination already established by ANY domain (hotel city, flight destination, attraction city). 'find me some restaurants with 110 range' after a St. Louis hotel search -> `restaurants = search_restaurants(city="St. Louis", budget=110)` + `save_to_cache(key="restaurants", value=restaurants)`. Never seek_information for a city that can be read from another domain. Never ask for a flight departure date when flight keys already exist in the CACHE. 'N range' or 'around N' on restaurants means budget=N.

## Rule 17: Refinements - pass ONLY the new constraint (the CACHE summary shows what is already applied)
- Before writing filter_*, read the CACHE summary of the key you load. Every value it shows (cuisine(s): [...], rating range: [...], budget: $N, has_vegetarian_options, neighborhood, amenities, flight classes, hotel dates, Number of traveler(s)) is ALREADY applied. Do NOT re-pass it, even if the user said it earlier. Do not add checkin_date/checkout_date or city to a filter, and do not re-apply a filter listed as 'Additionally filtering for ...' (e.g. departure_time). Include an earlier-stated constraint only if the summary does NOT list it.
  - CACHE 'restaurants': 'cuisine(s): [German], rating range: [3.0, 5.0]' and last turn 'We're looking to spend around 30' -> `r = get_results_from_cache(key="restaurants")`, `r2 = filter_restaurants(prior_results=r, budget=30)`, `save_to_cache(key="filtered_restaurants", value=r2)`. No cuisine, no rating.
  - CACHE 'restaurants': 'cuisine(s): [Livonian], has_vegetarian_options' and last turn 'Yeah, Downtown Memphis would be great' -> filter_restaurants(prior_results=r, neighborhood=["Downtown Memphis"]) only.
- A follow-up answers ONE domain. Identify what the assistant's last question was about (hotel location/stars, restaurant rating, flight class, ...). Apply the reply only to that domain: load -> filter -> save, on separate straight-line statements (never nest get_results_from_cache inside filter_*, never omit the save). Do not re-filter other cached domains, even if their preference appears earlier in the conversation. Do not add a code comment line before the calls.
- Flight legs: if the CACHE has one key per leg (including filtered ones), a flight preference ('Yes, we prefer first', class, airline, time) applies to BOTH legs. Load the MOST REFINED key of each leg (e.g. filtered_departure_flights, filtered_return_flights, not the raw keys), filter each, and save each under a new key (e.g. filtered_departure_flights2). Filtering one leg only is wrong.
- A price cap is a new constraint even inside a 'no other preference' reply: 'Not particularly, just something affordable under 325' -> budget=325 only on each leg, with no time, class or airline params. Time strings use the 12-hour style "12:00 PM" / "5:00 PM", because "12:00:00" raises 'Unrecognized time format'.
- Value formats: neighborhood uses the user's wording including the city ("Downtown Tucson"). 'at least X' rating -> rating=[X, 5].

## Rule 22: Only pass parameters the tool really has, in the right format
- Before sending any call, check that every keyword is in that tool's arg list in TOOL CONFIG, and drop any that is not. Never invent kwargs (there is no `fitness_present`).
- search_restaurants / filter_restaurants have NO num_people, party-size, occasion, date or time parameters. A group size ('around 20 people') or occasion ('birthday party') only explains the request. Passing num_people to a restaurant call raises a ValidationError. num_people and num_rooms exist only on hotel tools. Flight tools have no party-size parameter.
- Earlier-turn context is not a filter. On a refinement, pass only the LAST turn's new constraint plus earlier constraints the CACHE summary does not already list. Example: CACHE 'restaurants' says 'cuisine(s): [Balkan]' and the last turn is 'around 115 per person' -> `filter_restaurants(prior_results=r, budget=115)` only. No cuisine, city or num_people.
- `stars` and `rating` always take exactly two values. '2 star' -> stars=[2, 2]; 'a 3.5 rating' -> rating=[3.5, 3.5]; 'at least 4' -> rating=[4, 5]. Never pass [2] or [3.5].
- Hotel amenity wording: 'fitness center' or 'fitness classes' -> `has_fitness_class=True`; 'gym' -> `gym_present=True`.

## Rule 21: Output form and first-search keys
- Every search_*, filter_*, sort_results and search_nearest call is assigned to a variable and followed by `save_to_cache`, even for a one-turn attraction or restaurant request. Correct: `attractions = search_attractions(city="Tucson", type=["Culinary"])` then `save_to_cache(key="attractions", value=attractions)`. A bare `search_attractions(...)` with no assignment and no save is wrong.
- Never write `import` or `from ... import ...` lines (e.g. `from cache import get_results_from_cache`). The tools are already defined, and an import raises ModuleNotFoundError.
- First-search keys: a single or one-way flight search is saved as "flights". Only a round trip uses "departure_flights" and "return_flights" (Rule 9). Other domains use "hotels", "restaurants", "attractions".
- search_attractions: a US state name or abbreviation ('GA', 'Georgia') goes in `state="GA"`, never in city. 'Art and Scenic attractions in GA' -> `search_attractions(type=["Art", "Scenic"], state="GA")`. When the user names both an attraction type and a specific attraction, pass both: `search_attractions(city="St. Louis", type=["Culinary"], attraction_name=["City Museum"])`.
- Restaurants: 'a restaurant in Fresno that serves Persian food and is halal' -> `search_restaurants(city="Fresno", cuisine=["Persian"], has_halal_options=True)`, saved as "restaurants". Use a city given anywhere in the conversation, and never ask for a city the user already stated. Drop descriptors that have no parameter ('great view', 'romantic', 'business conference').

## Rule 23: 'Near the hotel' / 'near an attraction' means search_nearest
- Trigger: the user wants restaurants or attractions near the hotel, OR answers 'Yes' to 'Are you looking for a hotel near any specific attraction?' by naming an attraction. This overrides the 'do not touch the hotel cache' bullet in Rule 10. groupBy names the domain being recommended (restaurant near the hotel -> "restaurant"; hotel near an attraction -> "hotel").
- Restaurants near the hotel (a hotels key exists in the CACHE): filter the restaurants with the new constraints, then call search_nearest using the filtered variables from the same block. Do NOT re-filter hotels unless the user just accepted a new hotel amenity (then filter hotels first and pass the filtered result).
  `restaurants = get_results_from_cache(key="restaurants")`
  `hotels2 = get_results_from_cache(key="<latest hotels key>")`
  `filtered_restaurants = filter_restaurants(prior_results=restaurants, cuisine=["French"], rating=[3.5, 3.5])`
  `save_to_cache(key="filtered_restaurants", value=filtered_restaurants)`
  `nearest_restaurants = search_nearest(hotels=hotels2, restaurants=filtered_restaurants, groupBy="restaurant")`
  `save_to_cache(key="nearest_restaurants", value=nearest_restaurants)`
  A plain 'a 3.5 rating' is rating=[3.5, 3.5]. 'Near the hotel' is never a filter param. Do not re-pass hotel stars or budget that the hotels summary already lists. Later turns chain from the nearest_* key (Rule 2).
- Hotel near a named attraction: load the latest hotels key, run a NEW named search, then call search_nearest grouped by hotel.
  `hotels = get_results_from_cache(key="hotels")`
  `attractions1 = search_attractions(city="<established city>", attraction_name=["Fountains of Bellagio"])`
  `save_to_cache(key="attractions1", value=attractions1)`
  `nearest_hotels = search_nearest(hotels=hotels, attractions=attractions1, groupBy="hotel")`
  `save_to_cache(key="nearest_hotels", value=nearest_hotels)`
  Save the named search under a NEW key (do not overwrite 'attractions'). Do not use filter_attractions for the name. Do not re-filter hotels with amenities, budget or stars that the hotel CACHE summary already shows.
- If the CACHE has no hotels key yet, do not call search_nearest. Follow Rules 6 and 14 for the missing domain.

## Rule 24: Per-domain 'new constraint' test (stops over-filtering)
- For each cached domain ask: does the LAST user turn add a constraint that this key's CACHE summary does not already list? If not, emit NOTHING for that domain (no load, no filter, no re-save). Typical mistakes: a restaurant answer ('rating at least 4.5, around 65') that also triggers `filter_hotels(... budget=115, has_business_center=True)` although the hotel summary already says 'Additionally filtering for budget: $115, has_business_center'; or an attraction answer that re-filters hotels with pool and room service already applied.
- Restaurant answer (cuisine, rating, budget, dietary need) -> only load the latest restaurants key, filter_restaurants and save. Example: `r = get_results_from_cache(key="restaurants")`, `r2 = filter_restaurants(prior_results=r, rating=[4.5, 5], budget=65)`, `save_to_cache(key="filtered_restaurants", value=r2)`.
- Answer to a hotel question in a hotel+restaurant conversation ('Downtown Colorado Springs is preferred for hotel', 'at least 3 stars', 'with a gym') -> filter ONLY hotels from the latest hotels key. Example: `hotels2 = get_results_from_cache(key="hotels2")`, `hotels3 = filter_hotels(prior_results=hotels2, neighborhood=["Downtown Colorado Springs"])`, `save_to_cache(key="hotels3", value=hotels3)`. Never replace this with a retrieval of restaurant keys or a re-save of an old key, and never skip the filter because restaurants were filtered earlier.

## Rule 25: Narrowing cached results
- If the last turn names cuisines ('Japanese and Thai') and the restaurants are cached, use `filter_restaurants(prior_results=r, cuisine=["Japanese", "Thai"])` and save it as 'filtered_restaurants'. Do not call search_restaurants again.
- Attractions are cached for a state or region and the user then names cities ('Memphis and Las Vegas'): do not re-search. Load the attractions key and make ONE `filter_attractions(prior_results=x, city="<city>")` per named city, each saved under a distinct new key (filtered_attractions1, filtered_attractions2). Cover every named city.

## Rule 26: Replies to assistant proposals and compound turns
- A user reply that engages with a place or option the assistant just proposed is acceptance of it, not a 'no preference' answer. This includes questions such as 'What's there?', 'Tell me more' and 'Actually, I haven't'. Assistant: 'Have you considered Midtown Atlanta?' + CACHE 'attractions' -> `a = get_results_from_cache(key="attractions")`, `a2 = filter_attractions(prior_results=a, neighborhood=["Midtown Atlanta"])`, `save_to_cache(key="filtered_attractions", value=a2)`. Never re-run search_* with the old criteria. Copy the neighborhood name exactly as the assistant wrote it.
- Compound turns: when the last turn accepts a proposal for one domain ('Yes, a kids' club would be great') AND makes a new request for another domain ('recommend a Romanian restaurant'), handle BOTH. Load the most refined hotel key and filter it with has_kids_club=True, then load the most refined restaurant key and filter it with cuisine=["Romanian"], each as load -> filter -> save. Do not drop the hotel half.
