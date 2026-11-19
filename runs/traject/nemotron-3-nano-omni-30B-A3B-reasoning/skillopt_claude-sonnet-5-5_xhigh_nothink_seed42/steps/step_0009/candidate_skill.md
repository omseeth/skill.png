# Tool Calling Skill

## Output format (errors here score zero)
- `tool name` is the tool's full `### ` heading copied exactly, in the form `Provider: API` (e.g. `Arabic news API: Aljazeera`, `Easy Email Validation: Validate Email`, `Email API: Get Fake Email Address`). Never give only the provider or only the endpoint, even when the query names the tool loosely. The name is only the `### ` heading line (e.g. `Lost Ark: get stats`, `Epic Games Store - Free Games: Free Games`). Copy the complete heading character for character, including non-Latin script, hyphens, spaces, extra colons and any tail such as `?{parameters}`, `(basic)`, `/v1/weather` or `{type}` (e.g. `Arabic news API: Youm7 - اليوم السابع`, `Kubric: The Comprehensive Movie News API: Basic Search`, `dev-to-api: Get the month's top articles`; the heading is `National Weather Service: /alerts?{parameters}`, not `.../alerts`). Never copy the `Provider:` description line or any ` — tagline` text, and never put ` — ` in a tool name. The provider must be the one printed in that same heading: do not attach an endpoint to a sibling provider with a similar name (`Current conditions (detailed)` belongs to `Easy Weather`, not `RapidWeather`). Before answering, find each name verbatim as a `### ` heading in the tool list.
- One JSON object per call. Every object has exactly three keys, each appearing ONCE: `tool name`, `required parameters`, `optional parameters`. Always include all three, using `[]` when empty. Never put two calls in one object. Never repeat a key, because a repeated key silently overwrites the first list and loses values. Put ALL optional parameters for a call in the single `optional parameters` list.
- Each parameter is `{"name": "x", "value": "y"}`. Every value is a quoted string, including numbers (`"2023"`, `"1"`, `"9098084"`).

- Both lists contain ONLY `{"name": ..., "value": ...}` objects. Never write bare strings or repeat a name (wrong: `"required parameters": ["region"]`, `"required parameters": ["grade", "grade"]`; right: `[{"name": "grade", "value": "2"}]`). Even a single optional is an object inside the list: `"optional parameters": [{"name": "sort-by", "value": "popularity"}]`, never `["sort-by", "popularity"]`. Build each call as: tool name, then ONE `required parameters` list, then ONE `optional parameters` list holding every optional value. Wrong: `"optional parameters": [{"name": "composer", "value": "Bellman"}], "optional parameters": []`, because the second key erases the first.
- The answer is one flat, bracket-balanced list `[{...}, {...}]` inside `<answer>` tags. Do not close the list early or leave stray brackets.

## Parameters
- Put each parameter in the list where the chosen tool's schema puts it. Names under 'Required parameters' go in `required parameters` (even `page`-like fields). Names under 'Optional parameters' go in `optional parameters`, even when they carry the query's main input (e.g. `composer`). Use only parameter names from that tool's own schema. Do not borrow a similar name from a sibling tool (use `region` if the tool has `region`, not `loc`).
- For each call, check every qualifier in its clause (page, count, date bound, language, country or place, duration, unit, sort, category) against the tool's optional parameters. If one fits, include it, even though it is optional. 'Only include optionals when asked' still means include them whenever the query states or implies a value. Examples:
  - 'first page' -> page `1`; '10 per page' -> per_page `10`; 'first 20' -> pageSize `20`; '3 scores on page 1' -> per_page `3`, page `1`.
  - 'at least 5 minutes' -> min_duration `300` (seconds).
  - 'recorded after January 2017' -> from_record_date `2017-01-01`. Use the date field that matches the verb (recorded vs published).
  - 'subtitles in Hebrew' -> subtitle_lang `he`; 'US English' -> `en-US`.
  - 'stations in Indonesia' -> a keyword or filter parameter such as `keyword`=`Indonesia`, when the tool has one.
  - A named group or entity search -> the matching `by` field, when the schema offers one (e.g. `Group Name`).

- The subject of a search clause goes in the tool's text-filter optional (`title`, `name`, `q`, `keyword`) even when that field is optional: 'Batman games' -> title `batman`; 'Zelda games with Majora in the title' -> name `Majora`. 'First 20 results' or 'first page with 15 results' on a paged tool -> page `1` AND the size field (`pageSize` `20`). A platform word ('Steam giveaways') -> platform `steam`.
  - 'five 20-sided dice' -> dice `5`, sides `20`; 'draw 5 cards' -> pulls `5`; '3-day forecast' -> periods `3`; Fahrenheit and MPH -> tempunit `F`, windunit `MPH`; 'free PC game giveaways' -> platform `pc`, type `game`; 'US region, USD' -> the tool's region and currency fields; 'sorted by date' -> the sort field.
- Do not invent parameters the query gives no basis for (random-fact ranges, max, fragment, json, notfound). Never emit a parameter with an empty value.

## Tool selection
- Pick the endpoint whose name and description literally match the query's verb, object and input type, and tell siblings apart by what they do. 'Role account' -> the Role Account tool, not a disposable-domain checker. 'MX records' -> the MX tool. 'Disposable' -> a disposable checker. Several dice -> `Custom dice`, one die -> `Custom die`. A bare domain -> a domain-level tool, a full email -> an email-level tool. 'Details of a product' -> the detail tool, not search or autocomplete. Radio stations in a country -> the radio-stations tool, not a Spotify tool.
- Never pick a tool just because a parameter value fits (e.g. a country code). Prefer the tool whose parameters can express every constraint in the query (e.g. a forecast tool with day-count, temperature-unit and wind-unit params, and a location field that accepts the given ID format).
- If the query names a tool or provider ('using Easy Email Validation', 'the Email Existence Validator tool', 'the Email API'), use that provider's endpoint whose description matches the requested action ('check if disposable' -> `Email Existence Validator: Check for Disposable emails`; 'generate a fake business email' -> `Email API: Get Fake Email Address`).

## Which calls to make
- Make a short checklist of the query's requests. Each request gets one call from the single tool that best matches it, and every call must trace to a clause.
- N distinct emails, domains, IDs or entities for a one-value tool means N calls. An action the query explicitly requests twice (e.g. 'recheck Asos categories') means two identical calls. A clause like 'details of SKU X, including similar items' means two calls (the detail tool and the similar-items tool). The same entity needing different checks (domain availability vs disposable) means one call per check, each with its own matching tool.
- When one clause asks for two properties of the same input ('valid AND disposable'), choose the ONE tool whose description covers both. Do not stack several different validators on the same input 'to be thorough', and do not add a second tool for a check another clause already covers. Exception: the query names several distinct checks (e.g. deliverability, MX records and role account), which get one call each.
- Do not add 'helper', 'random', 'new releases' or 'list' calls that no clause names.
- Before answering, tick each sentence of the query (first and last included) against your calls. Add any missing call and delete any call that no clause asks for.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
