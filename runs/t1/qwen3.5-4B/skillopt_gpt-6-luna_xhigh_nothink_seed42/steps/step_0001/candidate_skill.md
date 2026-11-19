# Travel Planning Skill

## Learned Rules

- Before planning calls, check the tool schema for required inputs and use information already provided in the conversation. If required inputs for any requested domain are missing, make a single `seek_information` call asking only for those missing inputs. Defer searches and other planning calls until they are supplied. Do not invent values or assume defaults, such as a one-night hotel stay or checkout date.

## Reliable tool use and incremental travel planning
- Call provided tool functions directly; do not import, redefine, or wrap them. Keep generated code limited to the tool calls needed, and return it in the required format without extra commentary. Return `print("No planning needed")` only when cached results already satisfy the full current request and no further operation is needed.
- Read the full conversation and cache summaries. Carry forward relevant earlier details, pass explicitly stated values to the matching supported parameters, and do not invent preferences, optional values, or unrelated search domains. Preserve user-provided values; for example, map a state abbreviation to `state`, not `city`. For hotels, map “solo” to `num_people=1` and a stated party size to `num_people`; do not confuse guest count with room count.
- When a user adds or changes criteria, retrieve the most specific compatible cached result, including a previously filtered result when available. Apply new constraints to every affected result set, but do not reprocess unrelated domains or repeat work whose results already satisfy the request. If no compatible cache result is available, use supported constraints directly in the search rather than broad-searching and filtering unnecessarily. Use `search_nearest` for requests about proximity or access between result types.
- Save every newly generated search or transformation result—including filtered, sorted, or nearest results—under a unique cache key.
- For flights, search each explicitly requested leg separately, using its correct direction and date. Apply preferences that cover the whole trip to every affected leg, and reuse and filter compatible cached leg results when available. For multi-city itineraries, search only the specified legs; do not omit a provided leg or invent one.

- Use keyword arguments for tool calls except for `seek_information`.
