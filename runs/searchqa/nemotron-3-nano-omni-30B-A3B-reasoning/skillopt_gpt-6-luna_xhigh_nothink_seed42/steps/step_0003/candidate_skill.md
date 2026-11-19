# Question Answering Skill

## Clue-style and implicit questions
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording, then use the context to identify the entity or value that satisfies the clue.
- Track the clue's relationships carefully: distinguish a place from its country, a performer from a tribute act, and a quoted hint from its concrete referent. Do not answer with a nearby topic, merely repeat the clue, or respond True/False unless the prompt actually asks for a yes/no judgment.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity
- Give the shortest unambiguous answer at the requested granularity. When the context supplies a concise answer label, use that form rather than expanding it into a fuller name, title, or role description merely because the expansion is also correct. When a category is already specified in the question, provide the distinguishing value rather than repeating the category.
- For completion or compound clues, identify the missing shared component that makes the alternatives fit; do not repeat a clue-supplied component or return one full compound. For partial-name prompts, give only the requested unknown name part. Omit components already stated in the question unless they are needed for unambiguous identification.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to distinguish otherwise-confusable entities.

- Preserve grammatical number: use a plural answer when the clue asks for multiple members or refers to a plural set, rather than normalizing the answer to a singular form.

## Evidence checks

- Before answering, identify the type of thing requested (for example, a person, place, object, or category) and check that the candidate fits that slot. Prefer an explicit answer attached to the same clue over adjacent retrieval noise or a plausible association; do not substitute a related entity, example, remedy, or mechanism for the requested answer.
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.
