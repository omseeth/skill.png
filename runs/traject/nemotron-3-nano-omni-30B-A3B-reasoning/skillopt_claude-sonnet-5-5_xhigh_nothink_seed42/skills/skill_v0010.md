# Tool Calling Skill

## Output format (errors here score zero)
- `tool name` is the tool's full `### ` heading copied exactly, in the form `Provider: API` (e.g. `Arabic news API: Aljazeera`, `Easy Email Validation: Validate Email`, `Email API: Get Fake Email Address`). Never give only the provider or only the endpoint, even when the query names the tool loosely.
- One JSON object per call. Every object has exactly three keys, each appearing ONCE: `tool name`, `required parameters`, `optional parameters`. Always include all three, using `[]` when empty. Never put two calls in one object. Never repeat a key, because a repeated key silently overwrites the first list and loses values. Put ALL optional parameters for a call in the single `optional parameters` list.
- Each parameter is `{"name": "x", "value": "y"}`. Every value is a quoted string, including numbers (`"2023"`, `"1"`, `"9098084"`).
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
## STRATEGIC GUIDANCE (read before every answer)

### A. Build the JSON mechanically. Most failures come from broken structure.
1. Write each call as exactly one object: `{"tool name": "...", "required parameters": [...], "optional parameters": [...]}`. Immediately after the `optional parameters` list, write `}`. NEVER write a second `"optional parameters"` or `"tool name"` key in the same object. A repeated key silently erases the first one. Never add a trailing `"optional parameters": []` after a filled list.
2. The answer is `[` + obj + `, ` + obj + `, ` ... + `]`. Never put `]` between calls, never wrap calls in nested lists, and never end with `}}`. The first characters are `[{` and the last are `}]`.
3. Every object has all three keys. A call with no required params still carries `"required parameters": []`. Every parameter is an object `{"name": "x", "value": "y"}`. A bare string like `"page_id"` is invalid and gets dropped.
4. Before finishing, scan the answer once: count the objects, and confirm each has exactly one `tool name`, one `required parameters` and one `optional parameters`. Confirm the number of objects equals your checklist count.

### B. Tool names: always `Provider: Endpoint`
- Never use a name containing ` — ` or a description string. If a tool group is shown as `Provider — long description`, build the name from the provider plus the endpoint name listed under it (`Lost Ark: get raids by level`, `Card Draw Simulator: Cards`, `Pokedex: Get Specific Pokemon`, `Pokemon Unite Pokemons: List all Pokemons`, `Chess Puzzles: Puzzle by Id`, `Amazon Pricing and Product Info: Main Endpoint`).
- Queries often quote tool names loosely. Resolve each quote to the full heading with BOTH halves, for example `Zappos Realtime Data: Zappos product detail by Product ID` and `BestBuy Product Data: BestBuyProductData`. A provider-only or endpoint-only name scores zero.
- If the query explicitly names an endpoint (GetFutureEvents, GetPeopleByTopic, GetNewsByTopic, GetFAQsByTopic, checkYouTubeAuth, etc.), use that exact endpoint and its topic/page params. Do not substitute a sibling (Past vs Future, People vs PeopleByTopic).

### C. Do not drop calls. Enumerate before answering.
- Split the query into clauses and count every noun that is a separate fetch. 'Statistics, character classes, and the engraving system' is three calls (`Lost Ark: get stats`, `Lost Ark: get classes`, `Lost Ark: get engravings`). Also give separate calls for 'level 3 raids' and 'an item named X'.
- The LAST sentence ('Finally...') is the most commonly skipped. Example: 'mark price klines for ETHUSDT linear, 60-minute, between timestamps' is its own call: `Bybit: Get Mark Price Kline` with required category=`linear`, symbol, interval=`60`, start, end (timestamps exactly as given).
- A request repeated for two pages gives two calls with page `1` and page `2` and the stated page size.
- 'List of countries' on Flight Data_v2 means `Flight Data_v2: Data of countries in json format`, not the airport or routes endpoint. 'Routes' means `Data on the routes in json format`. These are separate calls.
- When the query lists several verification targets with no named tool, use a domain-parameter tool for a bare domain (`fast Email verifier: disposable Domain and Suggest Domain`, param `domain`). Use a dedicated email verifier for full addresses (`Email Address Validator: Validate Email Address` or `Alpha Email Verification: Email Checker`) rather than a generic disposable or 'Check Valid Emails' tool.

### D. Tool-selection reminders (evidence from failures)
- Forecast for a numeric location ID with N days, Fahrenheit and MPH means `Foreca Weather: Daily` with required `location`, and optional `periods`=`N`, `tempunit`=`F`, `windunit`=`MPH`. Do not use Tomorrow.io. Air pollution means `RapidWeather: Forecast air pollution data` (lat, lon). Active NWS alerts means `National Weather Service: /alerts/active/count`. Three requests means three separate objects.
- Dice: 'five 20-sided dice' means `Dice Roll Simulator: Custom dice` (dice=`5`, sides=`20`). The name is NOT `Custom dice rolls`. 'N regular dice, M times' means `Dice Roll Simulator: Regular dice rolls` (dice, rolls). 'Roll N regular dice' once means `Dice Roll Simulator: Regular dice` (dice).
- Cards: 'draw 5 cards' means `Card Draw Simulator: Cards` with optional pulls=`5`, suit=`all`, value=`all`, back=`false`. A specific card (e.g. an Ace) means `Card Draw Simulator: Card` with suit=`all` and value=`A`.

### E. Parameter value rules
- **Case.** Pokedex values are lowercase (`country`=`uk`, `pokemon`=`pikachu`). SBB Autocomplete query is lowercase (`zurich`). Keep other free-text queries and names as written (`Azumarill`, `Hollowfruit`, `iPhone 15 Pro`). Country codes for News API / Epic Games stay uppercase (`GB`, `US`, `CA`, `DE`).
- **Amazon.** `domain` is the marketplace suffix (`de`, `com`, `co.uk`), not `amazon.de`.
- **Named endpoints.** When the query names a function, e.g. 'Alpha Vantage's DIGITAL_CURRENCY_DAILY', the `function` value is exactly `DIGITAL_CURRENCY_DAILY`.
- **Placement.** `Stars API: Planet list` takes `starid` as an OPTIONAL parameter. Always check the schema's required/optional placement per tool rather than assuming.
- **BizToc.** `BizToc: News / Latest / Page` requires `page_id`; 'main page' means `page_id`=`main`.
- **TED Talks.** The date lower bound is `from_record_date` (`2017-01-01`), never `from_publish_date`. Speaker is a lowercase underscore slug (`yuval_noah_harari`). Include `topic`, `min_duration` (seconds) and `subtitle_lang` when stated, all inside ONE optional list.
- **Schema defaults.** For these locale/pagination-style tools, include the schema defaults even when the query is silent. For Asos: `country`=`US`, `lang`=`en-US`; `v2/auto-complete` also `currency`=`USD`, `sizeSchema`=`US`, `store`=`US`; `countries/list` only `lang`=`en-US`. For `Wayfair: products/search`: `curpage`=`1`, `itemsperpage`=`48`, `sortby`=`0`. Do not add defaults to Wayfair get-* tools or other tools.
- **Optional-only inputs.** Some tools take their main input in optional parameters (e.g. Chess Puzzles `id`, Pokemon Unite `page`/`pageSize`, Priceline `limit`). Put them in `optional parameters`.

### F. Keep doing what works
The passing answers all had one clean object per call, full `Provider: Endpoint` names, and explicit page, limit, country, pulls, dice and sides optionals, with each stated qualifier mapped to the schema's parameter name. Keep that discipline, and spend your remaining effort on completeness (every clause has a call) and structural cleanliness (Section A).
<!-- SLOW_UPDATE_END -->
