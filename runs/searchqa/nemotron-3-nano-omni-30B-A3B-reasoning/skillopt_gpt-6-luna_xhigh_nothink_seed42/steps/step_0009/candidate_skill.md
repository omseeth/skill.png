# Question Answering Skill

## Clue-style and implicit questions

- In trivia or wordplay clues, treat a parenthetical number as a hard answer-length constraint. For a double-definition clue (for example, “X, or what Y is put to”), seek the exact word that fits both senses rather than a synonym that fits only one. Use any proper-name cue to identify the requested name or name part.
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording, then use the context to identify the entity or value that satisfies the clue.

- When a clue asks about an abbreviation, expand it to the named feature rather than giving an associated mechanism or implementation detail. When a quotation visibly trails off, supply the next missing word if the clue asks for its continuation, rather than naming the work. For spoonerism clues, restore the intended phrase when that phrase is requested; do not substitute a person or other real-world referent.
- Track the clue's relationships carefully: distinguish a place from its country, a performer from a tribute act, and a quoted hint from its concrete referent. Do not answer with a nearby topic, merely repeat the clue, or respond True/False unless the prompt actually asks for a yes/no judgment.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity
- Give the shortest unambiguous answer at the requested granularity, using the concise form most directly supported by the clue and its evidence. Do not add an unnecessary corporate suffix or title subtitle, or repeat wording already supplied by the clue. For a person, use an explicitly supported full personal name rather than substituting a familiar nickname when the full name is the relevant answer. When a category is already specified, provide the distinguishing value rather than repeating the category.

- Omit nonessential expansions even when they appear in a fuller contextual label: prefer a brand's short public name over its expanded company name, and omit middle initials or ordinal/rank modifiers unless the clue requests them or they are needed to disambiguate. Preserve the requested title or name itself.
- For completion or compound clues, identify the missing shared component that makes the alternatives fit; do not repeat a clue-supplied component or return one full compound. For partial-name prompts, give only the requested unknown name part. Omit components already stated in the question unless they are needed for unambiguous identification.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to distinguish otherwise-confusable entities.

## Evidence checks

- For comparative and superlative clues, preserve the stated comparison dimension: for example, distinguish physical size from denomination or value, and compare candidates on the property the clue actually names.

- Before answering, identify the type of thing requested (for example, a person, place, object, or category) and check that the candidate fits that slot. Prefer an explicit answer attached to the same clue over adjacent retrieval noise or a plausible association; do not substitute a related entity, example, remedy, or mechanism for the requested answer.
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.

## Bare work-title prompts
- If a prompt consists only of a work title and year, do not answer with its generic type (such as “film”). Look for a principal person explicitly and centrally associated with that work in the context, such as its director, author, or creator, and answer with that person only when the association is clear. If the context offers no clear intended relation, treat the prompt as underdetermined rather than guessing.
