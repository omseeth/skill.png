# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.

- In declarative clue sentences, resolve personal or possessive pronouns to their antecedents rather than choosing a salient noun from a title. For a bare person-plus-office prompt, use context to infer the omitted relation (such as the country associated with the office) and return that value, not a biographical description.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For an explicit fill-in or name-completion prompt, normally supply only the missing headword or name component; however, when a blank is embedded in a recognizable multiword entity name and the clue asks you to identify that entity, give its full canonical name. Do not repeat information already given unless it is part of the canonical answer or needed to disambiguate. For people, use a full canonical personal name when the clue does not supply a name component, but do not add middle initials, middle names, or extra given names unless they are part of the supported canonical form or needed to distinguish candidates. For definition clues, give the requested category term rather than a list of examples, and match grammatical number and necessary articles. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

<!-- SLOW_UPDATE_START -->
Before answering, check that the response preserves the full identity and form requested by the clue. When context distinguishes a specific political entity from a broader country label, give the specific entity (for example, “People’s Republic of China,” not merely “China”). For people, give the full name rather than only a surname; include an honorific when the clue explicitly signals it, such as “knighted” requiring “Sir,” but do not add unsupported titles. For vocabulary clues, match the requested part of speech and inflection: if the clue asks for a term meaning the act or manner of following a winding course, use the corresponding form “meandering,” not the base verb “meander.”
<!-- SLOW_UPDATE_END -->
