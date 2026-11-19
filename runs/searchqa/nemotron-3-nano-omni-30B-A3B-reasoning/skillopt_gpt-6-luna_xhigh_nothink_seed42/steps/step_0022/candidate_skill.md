# Question Answering Skill

## Clue-style and implicit questions

- In trivia or wordplay clues, treat a parenthetical number as a hard answer-length constraint. For a double-definition clue (for example, “X, or what Y is put to”), seek the exact word that fits both senses rather than a synonym that fits only one. Use any proper-name cue to identify the requested name or name part.

- For a Spoonerism clue, identify the swapped sounds and restore them to recover the intended expression. Return that expression rather than the person or entity it refers to, unless the question explicitly asks for that entity.
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording and context rather than assuming the named entity itself is the answer. For a name-only officeholder prompt, consider an explicitly salient associated relation (such as a ruler and their realm); return that associated value only when the clue and context make the intended relation clear.

- When an ellipsis marks omitted words in a title or phrase, use context to recover and return the missing continuation. Prefer a completion that fits the whole fragment over a prominent work or entity that matches only a shorter substring.
- Track the clue's relationships carefully: distinguish a place from its country, a performer from a tribute act, and a quoted hint from its concrete referent. Do not answer with a nearby topic, merely repeat the clue, or respond True/False unless the prompt actually asks for a yes/no judgment.

- Resolve pronouns such as “they,” “this,” and “it” to their antecedents before interpreting nearby names. If a clue says something was named after a place, answer the thing being named when that is the requested referent, not the place supplying its name.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity
- Give the shortest unambiguous answer at the requested granularity, in the conventional lexical form and part of speech called for by the clue (for example, an adjective or demonym rather than its associated country). Prefer the exact context-supported term over a close synonym or related term, and preserve its canonical part of speech and number. Follow an explicit number cue when the clue asks for multiple entities or a class or group, including a plural answer where appropriate; do not force a singular label when a plural is requested or infer a plural merely from incidental plural wording when the canonical answer is singular. Only omit a category noun when it is truly redundant. Preserve a head noun that belongs to the conventional full proper name or is needed to express the requested common-noun answer; do not reduce such an answer to a modifier or genus alone. For class or taxon answers, use the conventional lexical number rather than blindly matching a nearby demonstrative such as “this.” Still omit explanatory labels and category words that the answer does not conventionally contain. For a person, use an explicitly supported full personal name rather than a familiar nickname when the full name is relevant, and preserve honorifics that are part of the context-supported conventional name, especially when the clue cues them.
- For completion or compound clues, identify the missing shared component that makes the alternatives fit; do not repeat a clue-supplied component or return one full compound. For partial-name prompts, give only the requested unknown name part. Omit components already stated in the question unless they are needed for unambiguous identification.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to distinguish otherwise-confusable entities.

## Evidence checks

- For comparative and superlative clues, preserve the stated comparison dimension: for example, distinguish physical size from denomination or value, and compare candidates on the property the clue actually names.

- Before answering, parse the requested answer slot and all identifying relations or properties, then test the candidate against the complete clue. When snippets compete, prefer factual evidence that fits those details; keyword overlap in a title or heading, prominence, or a similar name alone is not enough. Do not substitute a related entity, example, remedy, or mechanism for the requested answer.
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.

## Bare work-title prompts
- If a prompt consists only of a work title and year, do not answer with its generic type (such as “film”). Look for a principal person explicitly and centrally associated with that work in the context, such as its director, author, or creator, and answer with that person only when the association is clear. If the context offers no clear intended relation, treat the prompt as underdetermined rather than guessing.

<!-- SLOW_UPDATE_START -->
Before answering, check that the candidate is the kind of thing the clue asks for, not merely a related name or term. A factual trivia statement such as the clue about Wessex Poems calls for the person identified, Thomas Hardy—not True or False. A clue saying someone played two roles calls for the performer, not the works; a formula stating that an object's area is pi r squared identifies the circle, not its radius.

When a clue supplies a direct identity cue alongside a relational phrase, let the identity cue resolve the answer: for example, U.S. president, 1981–1989 identifies Ronald Reagan, not a related former leader. Use historical context to give the specific entity the clue identifies: the state admitted to the U.N. in 1971 is the People's Republic of China. Preserve a conventional honorific when supported by the clue or established name form; for Francis Drake, give Sir Francis Drake rather than Drake or Francis Drake.

Match the answer's specificity and grammatical form to the wording. If the clue asks for the event type, answer marathon rather than an expanded name for a particular race. If it asks for the term for following a winding course, give Meandering rather than the verb meander. Do not replace the requested form with a related root, variable, example, or more elaborate proper name.

Keep the successful concise approach on straightforward clues: verify that the answer fits the identifying details, then return only the requested person, place, object, or value.
<!-- SLOW_UPDATE_END -->
