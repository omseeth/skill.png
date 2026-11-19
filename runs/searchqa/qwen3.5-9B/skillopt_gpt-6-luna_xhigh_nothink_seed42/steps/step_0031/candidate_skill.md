# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For definition clues, match the grammatical form requested by the wording, including inflection, rather than defaulting to the dictionary lemma; give the requested category term and match grammatical number and necessary articles. For a lexical blank, give the missing word, but when the clue asks you to identify a fixed multiword expression or named entity, give its full conventional name even if a component appears in the prompt. When identifying a person, use the conventional given-and-family name rather than a surname alone; give only a missing name component when the prompt explicitly asks for that component. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

For clipped trivia or crossword prompts, distinguish the actual clue from category labels, page headings, and adjacent search-result text; solve the clue rather than repeating those labels. For a bare pair or list of people with no explicit relation, use direct context to identify their salient shared place, affiliation, or category instead of defaulting to a generic relationship.

<!-- SLOW_UPDATE_START -->
For terse person-and-office clues, infer the requested relation from context and return its value (such as the country), not a biographical description or office-and-dates restatement. For comparisons, evaluate the stated property and time scope literally: do not confuse smallest denomination or historical smallest with smallest physical size among currently minted coins. In technical clues, honor explicit categories such as acid versus alkali and reject nearby substances that do not fit the category. For linked media clues, trace the named role to the specific work or series asked for rather than choosing an adjacent title. Use the conventional name without unnecessary expansion: expand an unambiguous familiar nickname to the formal given name, but omit unrequested middle initials, generational suffixes, and honorifics. Conversely, retain an essential entity-type word in an answer naming a publication when needed, as in “Forbes magazine.”
<!-- SLOW_UPDATE_END -->
