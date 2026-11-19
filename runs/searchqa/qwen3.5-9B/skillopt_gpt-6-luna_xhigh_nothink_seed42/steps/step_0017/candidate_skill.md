# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For fill-in or clue-style prompts, supply only the missing headword or name component when that is the requested answer. If the blank is embedded in a fixed multiword label and the requested referent is the label as a whole, return its complete conventional form rather than mechanically omitting the part already shown. Do not add clue modifiers unless they are part of the canonical answer. For definition clues, give the requested category term rather than a list of examples. Use the conventional canonical headword or name form supported by the directly relevant clue or source, including established plural-form labels even when referring to one instrument or category; do not mechanically singularize or pluralize for agreement. Include an article only when grammatically required. Use a full canonical personal name when the clue does not supply a name component; when it supplies part of a name and asks for completion, give only the missing component. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
