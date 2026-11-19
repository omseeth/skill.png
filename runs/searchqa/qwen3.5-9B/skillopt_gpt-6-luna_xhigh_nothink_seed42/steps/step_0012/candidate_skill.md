# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.

- For a quoted fragment ending in an ellipsis or an incomplete title or lyric, look for the exact phrase completion in context and return the missing words or phrase—not an artist, author, or performer merely associated with it.
- When several answers seem plausible, prioritize candidates that fit the clue’s exact wording and requested relation and are directly supported by context. Consider clue-specific, idiomatic, or metaphorical meanings, and check that a candidate fits all parts of the clue rather than relying on a generic assumption or prominence alone.
- Return one concise canonical answer at exactly the requested level of specificity. For fill-in or clue-style prompts, supply only the missing headword or name component; do not repeat information already given in the clue or add clue modifiers unless they are part of the canonical answer. For definition clues, give the requested category term rather than a list of examples, and match grammatical number and necessary articles. Use the concise canonical answer form supported by the directly relevant clue or source, including a surname or compact geographic name when that is the established answer; do not expand it into a synonymous longer form. Use a full canonical personal name when the question asks for the person's identity and no shorter form is called for. When a clue supplies part of a name and asks for completion, give only the missing component. Avoid explanations and alternatives.

### Original and altered titles or wording
- When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Apply this only when the requested answer is a title: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. If the question asks which person, character, or entity was parodied, give that referent rather than the parody's title.
- For a quoted phrase changed or distorted through wordplay, follow whether the prompt asks for the literal utterance or the intended or source expression: give the altered wording when asked what was literally said, and the original or restored wording when asked for the source or intended phrase. If a clue presents a familiar wordplay distortion without asking what was literally uttered, give the intended expression rather than repeating the distortion.

## Evidence Cross-Checking
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

### Bare entity prompts
- When the prompt is only an entity name, use the task framing and context to infer the missing attribute, then return that attribute rather than an encyclopedia-style definition. Base the inference on a directly supported relation in the context; if no relation can be inferred, do not invent one.

### Canonical titles and labels
- For a named work, event, or concept, retain the full conventional title or label rather than shortening it to one of its components. Omit descriptive modifiers that are not part of the canonical name, and use the established spelling and spacing.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
