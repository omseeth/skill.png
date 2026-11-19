# Question Answering Skill

## Clue-style and implicit questions
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording, then use the context to identify the entity or value that satisfies the clue.
- Track the clue's relationships carefully: distinguish a place from its country, a performer from a tribute act, and a quoted hint from its concrete referent. Do not answer with a nearby topic, merely repeat the clue, or respond True/False unless the prompt actually asks for a yes/no judgment.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity
- Give the shortest unambiguous answer at the requested granularity. When a category is already specified in the question, provide the distinguishing value rather than redundantly repeating the category; omit extra descriptors and expansions that are not needed to identify the answer.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to disambiguate.

## Evidence checks
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.
