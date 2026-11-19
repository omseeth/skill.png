# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the answer slot before using the context: determine whether the clue asks for an entity, attribute, role, or relation, and return the value that fills that slot rather than a related entity. Parse relational wording carefully (for example, a clue about the university of a territory may ask for the territory, not the university). A person or entity introduced as an example (such as after “like”) may indicate a category or role rather than be the answer. Treat fragments as implicit questions and infer the missing value instead of narrating about the named subject; use a true/false answer only when explicitly requested.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise, standalone answer at exactly the requested level of specificity, not a full sentence or explanation. For fill-in or clue-style prompts, supply the missing value, but preserve every word that belongs to the established canonical name or title even if the clue also uses a generic category word. For definition clues, give the requested category term rather than a list of examples, and match grammatical number and necessary articles. For personal names, use the conventional form best supported by the context and requested specificity; do not automatically expand a surname-form answer into a full given name unless the fuller name is needed to identify the person or is requested. When a clue supplies part of a name and asks for completion, give only the missing component. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.
