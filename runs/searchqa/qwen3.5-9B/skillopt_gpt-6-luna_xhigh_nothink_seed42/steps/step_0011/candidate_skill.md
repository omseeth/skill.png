# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer

- Follow any explicitly specified output format or schema exactly, including required tags or wrappers; keep the final answer itself concise and limited to the requested response.
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For fill-in or clue-style prompts, supply only the missing headword or name component; do not repeat information already given in the clue or add clue modifiers unless they are part of the canonical answer. For definition clues, give the requested category term, not a list of examples. When a clue seeks a lexical headword or category, use its conventional dictionary form (usually singular), even if the clue describes it in the plural; use a plural only when the question explicitly requests plural instances or a set. Include necessary articles. Use a full canonical personal name when the clue does not supply a name component; when it supplies part of a name and asks for completion, give only the missing component. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking
- Use equivalent paraphrases to interpret evidence, but in the final answer prefer the canonical label used by the directly relevant clue or source over a synonym, especially for fill-in clues. Prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

### Terse fragments and names
- When a prompt is a telegraphic fragment rather than an explicit question, infer the omitted relation from the wording and named entities. Do not substitute the dominant topic of retrieved passages for the attribute the fragment points to.
- When a fragment gives a role or title together with a person's name, use context to identify the associated jurisdiction or organization if that is the requested relation. Return that entity directly rather than restating the person's role or tenure.
- Retain conventional honorifics or titles (such as “Sir”) when they are part of the canonical name in context or are made salient by the clue; do not omit them merely to shorten a personal name.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
