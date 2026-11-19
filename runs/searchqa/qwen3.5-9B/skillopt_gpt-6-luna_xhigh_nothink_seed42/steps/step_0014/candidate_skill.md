# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Extract the Requested Answer
- Identify the specific entity, attribute, or relation requested before answering. Treat clue-like statements or fragments as implicit questions and return the missing value, not a judgment about whether the statement is true. Give the named referent rather than a related person, description, or broad topic; use a true/false answer only when explicitly requested.
- For each candidate, identify the clue’s governing predicate and expected answer type; verify that the candidate answers that relation and satisfies every explicit modifier. Do not substitute an associated name or detail for the requested relation, infer an answer from nearby but tangential text, or complete fragmented snippets with speculation. Prefer direct, relation-matching evidence and reject candidates that contradict a clue constraint.
- Return one concise canonical answer at exactly the requested level of specificity. For fill-in or clue-style prompts, supply only the missing headword or name component; do not repeat information already given in the clue or add clue modifiers unless they are part of the canonical answer. For definition clues, give the requested category term rather than a list of examples, and match grammatical number and necessary articles. Use the answer form established by the clue or a directly relevant keyed source rather than defaulting to the fullest form: a clue may call for a surname or compact label even when a longer canonical name is known. Include full names or formal qualifiers when requested or needed to disambiguate the referent, and do not add a narrower subtype or campus unless the question asks for it. When a clue supplies part of a name and asks for completion, give only the missing component. Avoid explanations and alternatives.

### Original vs. altered titles
When a question describes a title as changed, adapted, or parodied, distinguish the original title from the altered title. Use the wording to determine which is requested: a reference to the original work or an author's title asks for the original; a question asking what the adaptation, parody, or short was called asks for the altered title. Do not mistake the described change for the requested answer.

## Evidence Cross-Checking

- If the requested referent is uniquely supported but a secondary clue detail is outdated, imprecise, or contradicted by context, answer the referent instead of turning the response into a fact-check. Correct or discuss the discrepancy only if the prompt asks for verification or the discrepancy affects which answer fits; do not disregard qualifiers that genuinely determine the target.
- Treat equivalent paraphrases as matches, prefer direct statements or definitions over tangential references, and ignore irrelevant distractors.
- Use complementary passages to confirm or disambiguate an answer when helpful. Follow the requested format and retain any qualifiers necessary to the answer.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
