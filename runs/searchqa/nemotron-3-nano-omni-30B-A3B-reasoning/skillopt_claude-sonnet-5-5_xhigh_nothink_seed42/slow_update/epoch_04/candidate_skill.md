# Question Answering Skill

## Rule 1: Questions are Jeopardy-style clues; always answer with the missing entity
- The 'question' is usually a trivia clue, a category-like fragment, or a statement with a hidden answer (e.g. 'In Dijon & Bordeaux', '"...on my mind"', '"X" is the Norwegian title of his drama'). It is rarely a literal question.
- NEVER answer 'Not answerable', 'True/False', 'Yes/No', or a description of the clue. Always output the specific entity, word, or name that the clue is asking for.
- If the clue is a fragment, work out what single thing connects it to the context. Examples: places plus language context -> the language or nationality; a phrase with '...' -> the missing word; 'his/her/this [work]' -> the person who created it. The context passages about the topic usually reveal the link.

- A clue that is only a list of fictional character names, roles or catchphrases ('Inspector Luger, Wojo, Fish') asks for the WORK they all belong to (the TV series, film, book or play: 'Barney Miller'). Do not answer with one actor, one character, or the list items. Pick the title that every passage pairs with all the names. If the list is of real people, answer with the team, state or other entity they share.
- A trailing number in parentheses, e.g. 'Flirtatious floozy Flanders (4)', is a crossword-style LETTER COUNT for the answer. The answer MUST have exactly that many letters (ignoring spaces), so reject any candidate that does not ('FLOOSIE' has 6 letters). Such clues often define one half of a famous name or phrase using the other half or a related word: 'floozy' + 'Flanders' -> 'Moll' (Moll Flanders), the 4-letter word that pairs with the clue word in the passage titles. Answer with only that missing word. If a bare number appears in parentheses, treat it only as this hint, not as part of the question.
- Double-meaning clues ('Term for an international match, or what its players are "put to"', 'X, or Y'): the answer is ONE word that satisfies BOTH halves ('the Test': a Test match; 'put to the test'). Check every candidate against both halves. A dictionary synonym that fits only one half ('trial') is wrong. When passages are noisy, use your own knowledge of sports terms and idioms.
- A bare date + place (or date + event) fragment such as 'February 6, 1911 in Tampico, Illinois' asks for the person (or named thing) that the passages repeatedly tie to exactly that date and place, here the person born there that day. Give the bare name ('Ronald Reagan'), not the event, the place or the date.
- Jokes, asides and filler in the clue ('(If I eat it, can I hear the ocean?)', 'Need help in this subject? Call up...') are decoration. Answer from the factual core plus the anchor term. When the clue uses a local or foreign word ('In Haiti, lambi is this big spiral-shelled sea snail'), look for the passage that glosses it ('locally referred to as lambi') and answer with the common English name (Conch).

- DEFINITION / GLOSS clues ('Term for ...', 'This phrase describes ...', 'It's in dictionaries as ...'): the answer is the single TERM being defined. Find the dictionary or encyclopedia passage whose title or headword ('X | Define X', 'X - definition of X') is glossed with the clue's own words. Answer with that headword in the form shown, keeping hyphens and keeping the plural if the clue describes a group. Drop the clue's category noun unless it is part of the standard name, and do not substitute a description or synonym.
- 'N-word' hints ('this 3-word phrase') work like a letter count: the answer must have exactly N words.

## Rule 3: Look for the clue repeated verbatim in the context
- ALWAYS scan every passage first for the clue's distinctive words repeated verbatim, before using your own knowledge. Passages often reproduce the clue from a trivia/Jeopardy archive (J! Archive, jeopardy question dumps on GitHub, flashcard sites, trivia pages), followed by the answer. Typical layouts: 'CLUE text | answer. right: ContestantName.' (the answer is the text between '|' and '. right:'/'wrong:'), 'CLUE | Madge.', 'CLUE, THE ATMOSPHERE', 'Q: ... A: the atmosphere', or flashcard front/back. The archived clue may be slightly garbled (missing accents or letters, parentheses, dropped words), so match on the distinctive words.
- When such a verbatim hit exists, COPY that answer in its own form and spelling, including a leading 'the' or 'Sir' if shown. Do not lengthen it with descriptors ('Madge', not 'Madge the Manicurist'). It OVERRIDES any guess from your own knowledge, other passages, forum comments, a contestant's wrong response ('What is gravity?'), or an unrelated page that merely mentions the topic. Before the answer tags, briefly quote the matched line.
- Otherwise, build the answer from passages that match several of the clue's details.

- The clue is often a near-copy of a sentence that appears in many context passages with one slot blanked ('Only the arrival of the Cunard liner ___ ...'). Find that sentence and take the word or name that fills the slot. A phrase repeated across independent passages is strong evidence.
- Archive lines may also be formatted 'CATEGORY | clue text | answer'. Take the answer that follows the clue, even if the whole context is one short passage.
- Archive snippets are often truncated right after the clue (e.g. '...THIS PLAYWRIGHT, DAVID...'). Treat the visible part as a strong hint and complete the answer from other passages repeating the same fact. Choose the entity that recurs across many passages together with the clue's other named entities (e.g. the title in 'Thunder Road (film)' pages). Do not talk yourself out of a well-supported entity.
- If the context offers several candidates (a different player in the same article, a film and its song), pick the one that satisfies ALL distinctive details: exact numbers, team, years, place. Do not choose one that fits only some.

## Rule 4: Exact-match formatting of the final answer
- Give only the core answer: no category or descriptor words ('vitamin C' -> 'C', 'U2 tribute band' -> 'U2', 'Ellie Goulding song' is wrong), no added articles, no explanation. Keep a leading article only when it is part of the proper title or name as it appears in the context (e.g. 'The New York Times', 'The Divine Comedy').
- Use the shortest standard form that names the entity, as a trivia answer key would. If the clue says 'this vitamin', answer with just the letter. If the clue names a group or band, give the group's name only. For people, prefer the form used in a verbatim answer from the context (Rule 3); otherwise use the name form most natural for the clue.
- Use the singular form for a generic single object ('Well', not 'wells'), but keep the plural for crops, commodities and mass categories ('Bananas') or when the clue says 'these'/'ones'.
- Give only the MISSING part. If a word from the clue is part of the answer phrase, do not repeat it ('A "hasty" dessert' -> 'pudding', not 'hasty pudding'; 'Bridge to' this place -> 'Terabithia'). If the clue says 'this manicurist / this king / this dessert', do not append that role word to the answer.

- DELETE-CLUE-WORDS TEST: before answering, strike from your candidate every specific word or phrase the clue already states, and keep only the remaining head. 'Bret Harte wrote about these exiles "of Poker Flat"' -> 'The Outcasts' (not 'The Outcasts of Poker Flat'). 'Robert La Follette called it the Progressive party, but it was nicknamed this' -> 'Bull Moose' (not 'Bull Moose Party'). 'This funeral structure' -> 'Pyre' (not 'funeral pyre', and not 'funeral boat', which echoes the clue's 'funeral' and ignores that the passages say 'funeral pyre').
- When the clue says only 'one of these' / 'this <generic noun>' and the answer is a common noun, give the plain common noun with no added modifiers: 'a back shaped like one of these' -> 'Balloon' (not 'hot air balloon'). Modifiers copied from passages (hot air, synthetic, funeral, speech) are usually wrong unless part of the standard proper name.
- Exception (head noun is part of the standard name): when the clue itself gives a generic type noun ('this empire', 'this gland', 'these bombs') and the bare modifier alone would not name the entity, keep the head noun in the form the passages use. 'this empire' -> 'Roman Empire' (not 'Roman'); 'this gland' -> 'pituitary gland'; 'one of these bombs' -> 'hydrogen bomb'; 'term for a leg cramp' -> 'Charley horse'. The same holds for named events, deals, treaties, wars and laws when the clue gives only a generic descriptor with the head word ('1853 purchase that brought the U.S. to its present area' -> 'Gadsden Purchase', not 'Gadsden'). Test: would a trivia answer key accept the bare word alone? If yes ('Sirius', 'Roanoke', 'Amazon', 'Yalta'), drop the type word. If not, keep the head noun. Pure category labels ('play', 'king', 'spacecraft', 'novel') are still dropped: 'this play' -> 'Henry V'. The 'do not repeat clue words' rule is for a separate modifier already stated in the clue ('hasty' + pudding).
- For titles, give the standard title without quotation marks and keep internal punctuation as written in the context ('M*A*S*H', not 'MASH film').
- For nickname clues about a person ('known as "Uncle Walter"'), answer with the real name as it appears in the context (Walter Cronkite), not the nickname. Still apply the person-name rule above.
- Put only the bare answer in the answer tags. Do not write a sentence, restate the clue, or add dates, qualifiers or alternatives. Brief reasoning belongs before the tags.

- Calibrate name length to a trivia answer key, using the short common form that the passages repeat. Drop corporate suffixes ('McCormick', not 'McCormick & Company'). Drop taglines or subtitles of shows and films when a short title is in common use ('Studio 60', not 'Studio 60 on the Sunset Strip'). Drop city, sponsor or year prefixes on a generic event when the clue says 'this <event>' ('the marathon', not 'Athens Classic Marathon'). Never cut a part that identifies the person: keep the regnal numeral of a king or of a play's title character ('Richard III', not 'Richard'; 'Henry V').

## Rule 2: Answer the referent, not a word from the clue
- When the clue says 'this country / this vitamin / this man / one of these ...', the answer is the thing that phrase stands for. Do not copy a proper noun or phrase already in the clue.
- Example pattern: a clue about a region 'in this country' wants the country, not the region. A clue about 'one of these "watering holes"' wants the real object being nicknamed, not the nickname.
- Ask: what category is the answer (country, person, vitamin, object, band), and what member of that category fits all the clue details?

- Match the TYPE of the slot. When the context mentions several related entities (a film and its theme song, a series and its columnist, a daughter and her mother, a team and its city), answer only the entity the category noun asks for. 'this film' -> the film, not its song; 'this city's Sabres' -> 'Buffalo', not 'Buffalo Sabres'; 'this playwright' -> the playwright, not the actress named beside him. Other named entities in the clue are anchors, never the answer.
- 'One of these <role/title>' ('the 14th one of these spiritual leaders') asks for the title or category noun itself (Dalai Lama), not the individual named in the clue.

- POSSESSIVE / OWNER clues: when the clue attaches a work, album, song or sub-part to 'he / she / his / her / this <org>'s' ('"X" is the last song on his 1983 album'; 'A & B are in this organization's Hall of Fame, founded in 1967'), the answer is the PERSON or ORGANIZATION that made or owns it, never the work, album or part named in the clue. A title already quoted in the clue is the wrong slot. Test the identifying detail (year, number) against the candidate and choose the body that has it, not a larger related body.

- 'The <Modifier> type / style / version / kind / variety of this' ('the Denver type of this', 'the Boston version of this', 'the Manhattan kind of this') names a standard compound whose head noun is the answer. Search the passages for '<Modifier> <noun>' (e.g. 'Egg White Denver Omelet', 'Denver omelet') and answer with the bare generic noun ('Omelet'), not the modifier and not a branded or themed product from the passages. A candidate must contain or fit the clue's distinctive modifier. A brand item such as 'Egg White Delight McMuffin' that matches only the side theme (cholesterol, egg whites) is wrong. Other clue details (diet, ingredient, cooking method) are just conditions on the generic dish.
- 'X needs A & one of these' / 'A, B and one of these': the clue already gives some members of a set. Find the passage listing the full set ('a subject and a predicate') and answer with the missing member ('subject'), never one already named in the clue.
- Clue lists its own options ('Of North Park, East Park, West Park or South Park, it's where...'): the answer is exactly one of the listed options, copied verbatim. Use the context to pick which.
- A word in quotation marks inside the clue (a 'drag', a nickname) is usually a pun or hint tied to the answer. Look for passages where that word appears next to a candidate and answer with that candidate.
- Anchor on the clue's most specific unique detail (a named person, capital, rare term such as 'Helots' or 'Khartoum') and choose the entity the passages repeatedly tie to it. When the context is clearly outdated on a superlative or descriptor ('largest', 'first'), let the unique detail win (Khartoum -> Sudan even if Sudan is no longer 'largest').
- Honor qualifiers when the context offers several names for the same thing: 'Greek god of erotic love' -> 'Eros' (not Roman 'Cupid').

<!-- SLOW_UPDATE_START -->
## Strategic guidance (read before every answer)

Most of your misses are NOT knowledge gaps. You usually land on the right area and then answer with the wrong slot or an over-long form. Examples of misses that must not repeat: 'Sultan of Brunei' (key: Brunei), 'Great Highland Bagpipe' (Bagpipes), 'Adirondack Mountains' (Adirondacks), 'Jean-Honore Fragonard' (Fragonard), 'George H. W. Bush' (George Bush), 'World Golf Hall of Fame' (LPGA), 'dugong' (manatee), 'Eric' (Rumsfeld), 'Show Boat' (Showboat).

Before the answer tags, write ONE line: `Type: __ | Constraints verified: __ | Trimmed form: __`. Do it in this order: (A) fix the slot type, (B) test the candidate against every hard detail, (C) run the trim pass. Keep what already works: copy verbatim archive answers (Rule 3) in their own form, and answer from your own knowledge when the passages are junk.

### A. Fix the slot type FIRST
- 'He / She / this man / this artist / this cabinet member / Who' -> a PERSON'S NAME. Never a film, song, place, or a bare first name ('Eric' is never acceptable). For a quotation attributed to an office, work out who held the office at that date and said it (apology to mistreated Iraqis -> the Defense Secretary during Abu Ghraib -> Rumsfeld).
- 'Title + personal name' alone ('Sultan Haji Hassanal Bolkiah', 'President Tran Duc Luong', 'Prime Minister X') -> the COUNTRY, as one bare word: 'Brunei', 'Vietnam'. Never 'Sultan of Brunei' or 'President of Vietnam'.
- 'X is in THIS ORGANIZATION'S Hall of Fame / has won this body's award / belongs to this body' -> the ORGANIZATION itself, as its standard acronym or short name ('LPGA'). A Hall of Fame, award or trophy named in or near the clue is an anchor. Use the year in the clue to pick the exact body (a Hall of Fame founded in 1967 is the LPGA's, not the broader World Golf Hall of Fame).
- 'The <Modifier> version / type / kind of this <instrument / dish / tool>' (e.g. 'The Highland version of this instrument has a blowpipe, a chanter & 3 drones') -> the bare generic noun ('Bagpipes'), not the modifier, not 'Great Highland Bagpipe'. Use the plural when the instrument or item is normally plural ('Bagpipes', 'Scissors').
- 'Relative of X, found in <place>' / 'cousin of X' -> a DIFFERENT entity from X and from any relative named in the clue. The place or habitat qualifier is a hard filter. Steller's sea cow's relative 'found in Florida' -> manatee (the dugong lives in the Indo-Pacific, so reject it).
- 'These mountains / islands / lakes' (a plural generic noun in the clue) -> the common plural short name WITHOUT the generic noun: 'Adirondacks', 'Rockies', 'Catskills', 'Andes'. Do not append 'Mountains'.
- A clue that is only a list of songs, characters or places -> the single WORK or entity that contains them all (songs 'Make Believe', 'Cotton Blossom', 'Can't Help Lovin' Dat Man' -> the musical Show Boat).
- Any content word or proper noun already in the clue is an anchor, never the answer. If your candidate appears in the clue, reject it and ask what the clue is describing.

### B. Verify EVERY hard detail with your own knowledge
- Test the candidate against the named team or organization, year, number, place or habitat, category word, and superlative dimension (size vs value). One failed detail means reject it, even if the passages favor it.
- 'During his career with the BREWERS...' -> a Brewers player (Robin Yount), not a Royals player. 'Very caustic ACID' -> an acid (hydrochloric acid). 'Smallest IN SIZE' -> the dime.
- Check the direction of events ('finally ADMITTED to the U.N. in 1971' -> China).
- When passages are generic, contradictory or off-topic, trust well-established knowledge. Never output a stray token from a passage.
- Captions and parentheticals ('seen here', '(U.S. president, '81-'89)', 'CATEGORY:clue') define the answer's role. The part before a colon describes the answer and the part after is the linked item.

### C. TRIM PASS (do this last, every time)
Most lost points are over-long answers. Strip until a quiz answer key would accept it.
1. PERSON NAMES:
   - Never include initials or middle initials: 'George Bush', not 'George H. W. Bush'. If the clue itself spells the name ('won the election... George Bush'), use that spelling.
   - Drop hyphenated or accented extra given names ('Fragonard', not 'Jean-Honore Fragonard').
   - When the clue's descriptor is a role or profession ('Artist whose most famous painting...', 'this cabinet member', 'this composer / painter / philosopher') and the person is known by surname, answer with the SURNAME alone: 'Fragonard', 'Rumsfeld'.
   - For biographical 'He/She...' clues about entertainers, frontiersmen, presidents, justices and similar, give the standard full name ('Johnny Cash', 'Daniel Boone', 'Charles Evans Hughes', 'Angela Bassett').
   - A verbatim archive answer (Rule 3) overrides all of this: copy its form.
   - Add 'Sir' only for people the clue calls knighted or who are conventionally titled.
2. Delete every word the clue already states, plus every role or category label ('Sultan', 'Hall of Fame', 'Mountains', 'Party', 'Conference') unless it is part of the standard name and a bare version would not be accepted.
   - Keep the head noun only for true standard names: pituitary gland, hydrochloric acid, Roman Empire, Gadsden Purchase, Charley horse, Smithsonian Institution.
   - Drop it when a bare name works: Yalta, Sirius, Adirondacks.
3. Match number and grammatical form: plural for 'these' or naturally plural things (Sails, Bagpipes), singular for a single object, and a noun for a noun definition.
4. Spelling of compound words: if any passage (especially an archive-style line) writes a title or compound as one word ('Showboat'), copy that exact form. Otherwise use the standard spelling.
5. If your answer has 3+ words or more than one proper-name part, look for the shorter standard form.

### D. Never give a non-answer
Always commit to one specific entity of the right type and form. Never answer with a description, a title-plus-country phrase, a first name alone, a clue word, or a wrong-type guess. If undecided between two candidates, take the one that satisfies the most clue details and has the right type.
<!-- SLOW_UPDATE_END -->
