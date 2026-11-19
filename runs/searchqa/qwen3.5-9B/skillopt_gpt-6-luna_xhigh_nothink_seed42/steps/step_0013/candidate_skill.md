# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For fill-in or clue-style prompts, supply only the missing headword or name component; do not repeat information already given in the clue or add clue modifiers unless they are part of the canonical answer. For definition or terminology clues, use the exact headword or category label directly defined in the most relevant context when available; do not substitute a more specific synonym or append explanatory words. Match grammatical number and include necessary articles. When asked who a person or character is, give the established full identity, retaining essential distinguishing suffixes such as regnal numbers or Roman numerals. If the clue supplies a name component and asks for completion, give only the missing component. Do not prepend a role or description unless requested. Avoid explanations and alternatives.

### Titles, adaptations, and title variants
- Distinguish an adaptation's title from its source work. If the question asks for the adaptation's title, give that title; if it asks for the original, source, or classic that an adaptation retells, give the source work—not the named adaptation. Apply the same distinction to parodies and altered titles.
- When context gives both an original-language title and a translated or expanded title, use the form explicitly requested or directly identified as original when appropriate. Do not combine title variants or add a translation or subtitle unless requested.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

### Resolve the target and its scope
For terse or fragmentary clues, identify the grammatical target and the kind of entity requested before choosing a candidate. Keep the answer at that same level: a city or site is not its famous landmark or an individual building within it. Treat supporting details as clues to the target, not as the answer, and use context to resolve abbreviations or ellipsis.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
