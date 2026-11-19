# Question Answering Skill

## Clue-style and implicit questions
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording, then use the context to identify the entity or value that satisfies the clue.
- Track the clue's relationships carefully: distinguish a place from its country, a performer from a tribute act, and a quoted hint from its concrete referent. Do not answer with a nearby topic, merely repeat the clue, or respond True/False unless the prompt actually asks for a yes/no judgment.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity
- Give the shortest unambiguous answer at the requested granularity. When the context supplies a concise answer label, use it rather than adding merely descriptive expansions. When a category is already specified, normally give the distinguishing value rather than restating the category; however, retain a category-like word when it is integral to the context-supported canonical proper name or title.
- For completion or compound clues, normally provide only the missing component; when alternatives share a component, give that component rather than one full alternative. For partial-name prompts, give only the requested unknown name part. However, when the prompt asks you to identify an established proper name, title, or other named entity—not merely fill a blank—use its context-supported canonical full form, including a clue-supplied component when needed to identify the referent. Otherwise, omit components already stated unless they are needed for unambiguous identification.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to distinguish otherwise-confusable entities.

## Evidence checks

- Before answering, identify the type of thing requested (for example, a person, place, object, or category) and check that the candidate fits that slot. Prefer an explicit answer attached to the same clue over adjacent retrieval noise or a plausible association; do not substitute a related entity, example, remedy, or mechanism for the requested answer.
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.

## Work-title clues
- For terse clues featuring a work title, do not assume the answer is the title itself or a principal person associated with it. Infer the intended relation from the clue and context: it may point to a source or work being retold, or to a creator, performer, character, or other clearly identified relation. Use an explicit contextual association when the clue makes that relation clear; if no intended relation can be established, treat the prompt as underdetermined rather than guessing.
