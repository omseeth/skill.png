# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer

- For a prompt consisting only of a work title and year, do not default to a broad category such as genre. Use the context to identify the salient named relation associated with the work (for example, its director or author) and answer with that entity; give a category only when the prompt asks for classification.
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For fill-in or clue-style prompts, supply only the missing headword or name component; do not repeat information already given in the clue or add clue modifiers unless they are part of the canonical answer. For definition clues, give the requested category term rather than a list of examples, and match grammatical number and necessary articles. For personal names, use the conventional identifying form supported by the clue and context, not automatically the fullest form found in a source. Include the given and family names needed to identify the person, but omit middle initials, honorifics, and generational suffixes unless requested or needed to distinguish people. When a clue supplies part of a name and asks for completion, give only the missing component. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

### Distinguishing formal entity names
When direct evidence distinguishes related political entities or governments by their formal names, preserve the distinguishing words in the answer rather than shortening the name to a potentially ambiguous common label.
