# Question Answering Skill

## Clue-style and implicit questions

- In trivia or wordplay clues, treat a parenthetical number as a hard answer-length constraint. For a double-definition clue (for example, “X, or what Y is put to”), seek the exact word that fits both senses rather than a synonym that fits only one. Use any proper-name cue to identify the requested name or name part.

- For a Spoonerism clue, identify the swapped sounds and restore them to recover the intended expression. Return that expression rather than the person or entity it refers to, unless the question explicitly asks for that entity.
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording, then use the context to identify the entity or value that satisfies the clue.

- When an ellipsis marks omitted words in a title or phrase, use context to recover and return the missing continuation. Prefer a completion that fits the whole fragment over a prominent work or entity that matches only a shorter substring.
- Track the clue's relationships carefully: distinguish a place from its country, a performer from a tribute act, and a quoted hint from its concrete referent. Do not answer with a nearby topic, merely repeat the clue, or respond True/False unless the prompt actually asks for a yes/no judgment.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity
- Give the shortest unambiguous answer at the requested granularity, in the conventional lexical form and part of speech called for by the clue (for example, an adjective or demonym rather than its associated country). Prefer the exact context-supported term over a close synonym or related term, and preserve its canonical part of speech and number. Follow an explicit number cue when the clue asks for multiple entities or a class or group, including a plural answer where appropriate; do not force a singular label when a plural is requested or infer a plural merely from incidental plural wording when the canonical answer is singular. When the clue already supplies the category or type, return the distinguishing value alone: do not append a generic category noun, institution label, explanatory wording, or unnecessary title, subtitle, or suffix unless needed for disambiguation. For a person, use an explicitly supported full personal name rather than a familiar nickname when the full name is relevant, and preserve honorifics that are part of the context-supported conventional name, especially when the clue cues them.
- For completion or compound clues, identify the missing shared component that makes the alternatives fit; do not repeat a clue-supplied component or return one full compound. For partial-name prompts, give only the requested unknown name part. Omit components already stated in the question unless they are needed for unambiguous identification.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to distinguish otherwise-confusable entities.

## Evidence checks

- For comparative and superlative clues, preserve the stated comparison dimension: for example, distinguish physical size from denomination or value, and compare candidates on the property the clue actually names.

- Before answering, identify the exact referent requested, resolve demonstratives and antecedents, and determine which side of each relation the answer occupies. Test a candidate by substituting it into the clue: an associated organization is not its constituent business class, a building within a site is not the site, and a specific example or product is not a requested general type. Verify every distinctive detail and prefer direct evidence for the full relation over keyword overlap, prominence, or a nearby related passage.
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.

## Bare work-title prompts
- If a prompt consists only of a work title and year, do not answer with its generic type (such as “film”). Look for a principal person explicitly and centrally associated with that work in the context, such as its director, author, or creator, and answer with that person only when the association is clear. If the context offers no clear intended relation, treat the prompt as underdetermined rather than guessing.

<!-- SLOW_UPDATE_START -->
Before answering, check that the candidate is the kind of thing the clue asks for, not merely a related name or term. A factual trivia statement such as the clue about Wessex Poems calls for the person identified, Thomas Hardy—not True or False. A clue saying someone played two roles calls for the performer, not the works; a formula stating that an object's area is pi r squared identifies the circle, not its radius.

When a clue supplies a direct identity cue alongside a relational phrase, let the identity cue resolve the answer: for example, U.S. president, 1981–1989 identifies Ronald Reagan, not a related former leader. Use historical context to give the specific entity the clue identifies: the state admitted to the U.N. in 1971 is the People's Republic of China. Preserve a conventional honorific when supported by the clue or established name form; for Francis Drake, give Sir Francis Drake rather than Drake or Francis Drake.

Match the answer's specificity and grammatical form to the wording. If the clue asks for the event type, answer marathon rather than an expanded name for a particular race. If it asks for the term for following a winding course, give Meandering rather than the verb meander. Do not replace the requested form with a related root, variable, example, or more elaborate proper name.

Keep the successful concise approach on straightforward clues: verify that the answer fits the identifying details, then return only the requested person, place, object, or value.
<!-- SLOW_UPDATE_END -->
