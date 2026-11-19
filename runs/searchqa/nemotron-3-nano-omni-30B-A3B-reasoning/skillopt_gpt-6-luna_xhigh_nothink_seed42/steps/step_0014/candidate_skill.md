# Question Answering Skill

## Clue-style and implicit questions

- In trivia or wordplay clues, treat a parenthetical number as a hard answer-length constraint. For a double-definition clue (for example, “X, or what Y is put to”), seek the exact word that fits both senses rather than a synonym that fits only one. Use any proper-name cue to identify the requested name or name part.

- For a Spoonerism clue, identify the swapped sounds and restore them to recover the intended expression. Return that expression rather than the person or entity it refers to, unless the question explicitly asks for that entity.
- Treat terse statements, quoted fragments, and incomplete phrases as information-seeking clues when appropriate. Infer the answer slot from the wording, then use the context to identify the entity or value that satisfies the clue.

- When an ellipsis marks omitted words in a title or phrase, use context to recover and return the missing continuation. Prefer a completion that fits the whole fragment over a prominent work or entity that matches only a shorter substring.
- Determine the answer's semantic type and referent from the clue's grammar before choosing among context entities. Follow pronouns to their antecedents, and distinguish the requested entity from its origin or location, an associated institution, or a related owner or property. Treat the clue's head noun and explicit type or modifiers as constraints (for example, acid versus alkali); a shared keyword or broad role is not enough. Use nearby entities as evidence, not as answers unless the clue asks for them. Do not merely repeat the clue or respond True/False unless the prompt asks for a yes/no judgment.
- Do not default to Not answerable just because the prompt lacks a question word. Use reasonable contextual inference when supported by the evidence; abstain only when the answer genuinely cannot be determined.

## Exact answer granularity
- Give the shortest unambiguous answer at the requested granularity, in the conventional lexical form and part of speech called for by the clue (for example, an adjective or demonym rather than its associated country). Prefer the exact context-supported term over a close synonym or related term, and preserve its canonical part of speech and number. Follow an explicit number cue when the clue asks for multiple entities or a class or group, including a plural answer where appropriate; do not force a singular label when a plural is requested or infer a plural merely from incidental plural wording when the canonical answer is singular. When the clue already supplies the category or type, return the distinguishing value alone: do not append a generic category noun, institution label, explanatory wording, or unnecessary title, subtitle, or suffix unless needed for disambiguation. For a person, use an explicitly supported full personal name rather than a familiar nickname when the full name is relevant, and preserve honorifics that are part of the context-supported conventional name, especially when the clue cues them.
- For completion or compound clues, identify the missing shared component that makes the alternatives fit; do not repeat a clue-supplied component or return one full compound. For partial-name prompts, give only the requested unknown name part. Omit components already stated in the question unless they are needed for unambiguous identification.
- Use the context-supported conventional form of a name or answer, and include additional name parts only when needed to distinguish otherwise-confusable entities.

- The shortest answer must still be semantically complete: retain a needed head noun when a bare genus or modifier does not name the requested item, and retain a qualifier when it identifies the specific entity or polity intended. Omit category words only when the remaining answer is a complete conventional form at the requested granularity.

## Evidence checks

- For comparative and superlative clues, preserve the stated comparison dimension: for example, distinguish physical size from denomination or value, and compare candidates on the property the clue actually names.

- Before answering, parse the requested answer slot and all identifying relations or properties, then test the candidate against the complete clue. When snippets compete, prefer factual evidence that fits those details; keyword overlap in a title or heading, prominence, or a similar name alone is not enough. Do not substitute a related entity, example, remedy, or mechanism for the requested answer.
- Allow for paraphrases when matching clues to context. For multi-detail clues, verify that a candidate fits the identifying attributes; corroborate it with other relevant passages when available, and disregard unrelated retrieval noise.

## Bare work-title prompts
- If a prompt consists only of a work title and year, do not answer with its generic type (such as “film”). Look for a principal person explicitly and centrally associated with that work in the context, such as its director, author, or creator, and answer with that person only when the association is clear. If the context offers no clear intended relation, treat the prompt as underdetermined rather than guessing.

## Direct clue-answer evidence
- When context contains an explicit answer paired with the same or a near-identical clue, treat that answer as primary evidence and preserve its form and granularity. Do not expand it using fuller names or labels from other passages unless the question requires that added detail for disambiguation.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
