# Question Answering Skill

## Clue-style and implicit questions
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording, then use the context to identify the entity or value that satisfies the clue.

- For a declarative clue, first turn it into a direct one-slot question (for example, “Who is described?” or “What thing has this property?”). Check that the answer fills that slot, rather than naming a component, associated item, or other entity mentioned nearby.
- Track the clue's relationships carefully: distinguish a place from its country, a performer from a tribute act, and a quoted hint from its concrete referent. Do not answer with a nearby topic, merely repeat the clue, or respond True/False unless the prompt actually asks for a yes/no judgment.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity

- For definition-style clues asking for a common term or a member of a broad class, answer with the minimal headword or class at the clue's level. Do not add a narrower named entity or context-derived modifier unless the clue requests that specificity or it is needed to distinguish the answer.
- Give the shortest unambiguous answer at the requested granularity. When the context supplies a concise answer label, use that form rather than expanding it into a fuller name, title, or role description merely because the expansion is also correct. When a category is already specified in the question, provide the distinguishing value rather than repeating the category.
- For completion or compound clues, identify the missing shared component that makes the alternatives fit; do not repeat a clue-supplied component or return one full compound. For partial-name prompts, give only the requested unknown name part. Omit components already stated in the question unless they are needed for unambiguous identification.

- When a prompt is just a comma-separated list of entities, check whether it is a clue for a shared attribute or relationship. If the context supports one, give that common answer rather than selecting one listed entity.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to distinguish otherwise-confusable entities.

- When the question and context use variant spellings, preserve the spelling used by the question or clearest relevant passage. Do not introduce diacritics or other spelling variants unsupported by those sources.

## Evidence checks

- Before answering, identify the type of thing requested (for example, a person, place, object, or category) and check that the candidate fits that slot. Prefer an explicit answer attached to the same clue over adjacent retrieval noise or a plausible association; do not substitute a related entity, example, remedy, or mechanism for the requested answer.
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.

## Bare work-title prompts
- If a prompt consists only of a work title and year, do not answer with its generic type (such as “film”). Look for a principal person explicitly and centrally associated with that work in the context, such as its director, author, or creator, and answer with that person only when the association is clear. If the context offers no clear intended relation, treat the prompt as underdetermined rather than guessing.
