# Tool Calling Skill

## Rule 1: Output contract (check before answering)
- `tool name` must be copied character-for-character from the tool's `### ` heading, in the form `Provider: API` (e.g. `BestBuy Product Data: BestBuyProductData`, `Airport Guide - Aviation Info: Airline by Country`). Never shorten it to only the endpoint or only the provider, and never re-space, rename or paraphrase it.
- Every call has exactly three keys: `tool name`, `required parameters`, `optional parameters`. Use `[]` when empty. Do not add other keys and do not repeat a key.
- Every parameter is an object `{"name": "x", "value": "y"}`. Never emit bare names such as `["all"]`. Write every value as a quoted string, including numbers (`"16"`, `"1"`, `"true"`).
- The answer is ONE flat JSON list `[{...}, {...}]` inside the `<answer>` tags. Do not close the list early and leave no stray `]` or `}` between objects. Check that brackets are balanced.
- Make one call per distinct item (one per ID, email, domain, user, or entity) unless the tool explicitly accepts a list.

## Rule 2: Cover every part of the query (build a checklist first)
- Before answering, split the query into atomic requests: each sentence, each 'then / also / and / finally / additionally' clause, each numbered item, each aspect in a list (e.g. 'pricing, warranty, reviews, and product info' = 4 calls), and narrative past-tense steps ('was looked up', 'was accessed'). All of these count as actions. Hedges such as 'possibly' or 'potentially' do not make an action optional, and neither does a phrase like 'less relevant'.
- Each atomic request needs its own call. A phrase like "today's X insight AND information about Y" means two calls: the 'today/daily' endpoint and a separate topic/search endpoint taking Y.
- The same endpoint may be called several times with different arguments when the query names several entities (two users, two search terms, several domains).
- Count calls against the checklist before finalizing. Dropping a clause is the most common failure, so do not stop after the first few actions.
- If a clause names a tool that is not in the list, map it to the closest tool by function. Do not substitute a tool with a different meaning (e.g. past events for 'upcoming events'), and do not skip the clause.
- Do NOT add calls that no clause asks for: no 'helper' calls (languages list, search-then-get), and no call invented because an identifier looks like another tool's parameter. If one tool fully covers a request, do not call a second tool for it.

## Rule 3: Choose the most specific endpoint
- Query phrases usually paraphrase one specific tool's name or description. Pick the endpoint whose name and description literally mirror the query's verb, object and input type. Use a generic 'Search' endpoint only when no dedicated endpoint fits.
  - 'Playlists from user X' -> a 'Get user Playlists' style endpoint taking user_id. A named chart, list, or radical/stroke lookup -> its dedicated endpoint. 'Download location data' -> the 'Download ...' endpoint.
  - Input type decides the tool. A domain needs a domain-level tool and a full email needs an email-level tool. A product or item ID needs a 'by ID' or detail tool, not a keyword search. Do not pass an ID where a URL is required, and do not use an artist or playlist endpoint for an album ID.
  - Match the verb. 'Disposable', 'MX records', 'verify / deliverable', 'role account', 'format + disposable' (one tool whose description covers both) and 'help documentation' each point to a different tool. Do not reuse one generic validator for all of them.
  - A request for items from a named source with filters (count, sentiment) -> the listing endpoint that has those filters, not a keyword search where the source name becomes a search string. General outlets -> a general news-by-source tool, not a topic-specific tool that merely has a 'source' parameter.
- If the query names a provider or tool (BestBuy, Wayfair, Asos, Spotify, Blaze Verify), use that provider's endpoint. If it names several endpoints of one provider, call each separately. Otherwise prefer a same-provider tool for a sub-task over a generic tool from another provider.
- Distinguish sibling endpoints by quantity or variant semantics: plural 'dice' with a count -> 'Regular dice'; a single die -> 'Regular die'; 'N dice rolled M times' -> 'Regular dice rolls'; a single card -> 'Card'; several cards -> 'Cards'.
- When the query says the same platform was used in 'different approaches' or 'two separate searches' and the list has sibling endpoints (e.g. 'Item Search', 'Store Item Search 2', 'Item Search 3'), call each distinct sibling once rather than repeating one. Likewise, when several providers expose near-identical endpoints and the query lists several items, spread the items across the variants whose description fits the input.
- Prefer the endpoint that directly covers the request over a 'count' or admin-helper variant (e.g. 'hotels in <state>' -> a hotel listing endpoint). Prefer tools whose required parameters can be filled from the query text alone.
- When the query asks for news or info on a broad topic or region and the list has dedicated feed, list or sources endpoints whose topic clearly matches (e.g. a crypto feed for 'blockchain', regional-language feeds for a named region), include them alongside the main search call. Never include feeds for unrelated topics or regions.

## Rule 4: Parameters: placement, constraints, and value formats
- Follow the chosen tool's own schema exactly. Use ONLY parameter names listed for that tool. Sibling tools often name the same idea differently (startPrice/endPrice vs minPrice/maxPrice, region vs loc).
- A parameter listed under 'Required parameters' goes in `required parameters`. A parameter listed under 'Optional parameters' goes in `optional parameters`, even when the query supplies it as the main input (flight number, airport code, star id, store id, country, `q`). A tool whose required list is 'none' has an empty required list. Fill a value for every required parameter.
- Copy IDs, SKUs, ASINs, order numbers, postal codes, URLs, emails, domains, timestamps, words, search terms and numbers verbatim from the query. Never alter or guess them.
- Map EVERY constraint in the query to a parameter of that tool when one exists, and do not add parameters the query never implies. Examples:
  - counts and sizes: 'top 10' -> range '1-10'; 'up to 15 articles' -> pageSize/limit '15'; '20 posts' -> per_page '20'; 'three dice' -> dice '3'; 'recent N' -> num 'N'
  - paging and sorting: 'first page' -> page '1'; 'default sorting' -> sortby '0'; 'most helpful' -> HELPFUL; 'best match' -> BEST_MATCH; 'highest ratings' -> RATING_DESCENDING
  - filters: 'positive' -> sentiment; 'article' -> type; a price range -> the tool's start/end price fields; a colour -> the filters param, in the format its description shows
  - locale: country, region, or language ('from the UAE in English', 'US market', 'US English' -> en-US, 'British English' -> en-GB, 'UK market' -> country GB)
  - other: 'complete / all info' -> all=true; a specific card ('Ace') -> value=A
  - Keep topic words such as 'for beginners' inside the search text, and keep the query's search keywords unchanged.
- Use enumerated options, codes and formats exactly as the parameter description gives them. If options are listed (e.g. country codes AE, US), copy them exactly. Follow format hints such as domains like 'site.com', state abbreviations, and comma-separated lists. For free-text code-like fields with no listed options, use the short lowercase code (e.g. 'de').
- For hierarchical location fields (country/state/admin1/admin2/city), follow each field's description and put each level in the matching field.
- Copy entity names and values from the query. Never invent values such as dates, ids, dice sides, or placeholders like YOUR_API_KEY. If a required value is truly unspecified, use a minimal neutral value (an empty string or 'demo').
- For 'a random X' requests on word or list APIs, an amount/limit of '1' is reasonable when the tool offers one.

## Rule 5: Fan-out, repeats, implied values, optional hygiene
- If a tool takes one value (one email, domain, SKU) and the query lists N of them, emit N separate calls, one per value. Do not switch to a batch tool unless the query names it. Keep per-call constants (e.g. username=info) on every fanned-out call.
- If the query explicitly asks for the same action twice (e.g. 'similar items' mentioned early and again at the end), emit the identical call twice. Do not de-duplicate explicit requests, but never add hedging near-duplicate calls the query did not ask for. Put all stated parameters for a clause in a single call.
- Ignore look-alike distractors the query never mentions: account/inbox/temp-mail tools, bulk/file-upload tools, or search/list tools when an ID is already given.
- Capture implied values as well as stated ones. A subject such as 'Batman games' goes into the title/keyword/query parameter, and a country or language implied by the query goes into its matching parameter in the same call. Source, category or filter phrases ('from CNN', "the 'toysgames' category", 'including STEREO_A and STEREO_B') map to the matching string parameter, with multiple items joined by commas when the parameter is comma-separated. Position phrases ('starting from the first entry (offset 0)') map to offset/page.
- Include an optional parameter only when the query states or clearly implies a value for it. Do not add defaults, format flags or toggles (fragment, json, verbose, callback, start=0) that the query never mentions, and never emit an optional parameter with an empty or placeholder value.
