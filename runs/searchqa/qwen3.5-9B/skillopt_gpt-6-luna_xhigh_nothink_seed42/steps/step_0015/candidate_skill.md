# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For fill-in or clue-style prompts, supply only the missing headword or name component; do not repeat information already given in the clue or add clue modifiers unless they are part of the canonical answer. For definition clues, give the requested category term rather than a list of examples, and match grammatical number and necessary articles. When the question or directly relevant source gives an unambiguous form of a person's name, preserve that form rather than adding unrequested initials, suffixes, or expansions; retain titles or name components made salient by the clue or needed to distinguish the person. Otherwise use the canonical full name. When a clue supplies a name component and asks for completion, give only the missing component. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

### Prompts with an implicit shared relation
- When a prompt consists of a comma-separated list of entities, use context to identify the attribute or relation they share, such as a common country, jurisdiction, or organization. Return that shared entity rather than an incidental relationship between the listed people, unless the prompt cues that relationship.
- For a standalone quoted name or phrase, use direct contextual evidence to resolve the named work, source, or series it is associated with when that is the implied lookup. Give the associated entity, not a generic description of the phrase.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
