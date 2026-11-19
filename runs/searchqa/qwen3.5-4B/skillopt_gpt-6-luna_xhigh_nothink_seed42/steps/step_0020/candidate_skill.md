# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

For definition-style clues, match the requested part of speech, grammatical form, and number. Do not automatically reduce an activity or process noun to its base verb: the clue may call for a gerund or another inflected form. Conversely, a singular term or entity calls for a singular headword even when examples or supporting evidence are plural. Use the lemma only when the clue asks for it, and do not copy the number of nearby examples.

## Treat Clue Echoes as Search Cues
In noisy or search-style context, a passage may repeat the clue without supplying its answer. Treat that echo as a locator for relevant material, not as evidence for the missing answer; use substantive passages to identify the target and verify distinctive clue details. If the passage explicitly pairs the clue with an answer, it can serve as direct evidence.

When sources disagree on a secondary detail, separate that discrepancy from the identity-defining clues. Prefer a candidate independently supported by multiple distinctive details, but do not dismiss a contradiction if it changes which candidate fits.

## Preserve Full Canonical Geographic Names
When a geographic proper name conventionally includes a generic component such as “Lake,” “Mount,” or “River,” retain that component in the answer. Do not shorten the name to its modifier unless the context or clue establishes the shortened form as the requested label.

## Verify the Clue's Exact Relation

When a clue includes a distinctive quotation, epithet, or slogan, use it as an identifying cue and determine the relation requested: the entity it describes, the person who said or coined it, or the referent it alludes to. Follow the clue's wording and contextual attribution rather than returning the quoted phrase itself.
For clues linking entities, identify the precise relation and the role of the missing answer before selecting a candidate—for example, a source versus what an instrument measures, or a city represented in an acronym versus the location of a related institution. A candidate's name or topic appearing nearby is not enough: prefer evidence that supports the same relation and the clue's distinguishing details.

## Use the Conventional Name at the Requested Scope
For proper names and titles, use the conventional form that identifies the requested entity. Do not substitute a nickname or later brand name for the historically relevant label, omit words from a work's title when the work is requested, or add optional middle names. Conversely, when the clue requests a generic class, do not embellish it with a narrower associated name.

## Check for Shared Completions in Parallel Clues
When a clue presents alternatives or examples in parallel, check whether one word completes or classifies them both. The target may be that shared word rather than a full phrase inferred from just one part of the clue. Return the minimal shared completion unless the clue asks for a full phrase.

## Respect Answer-Length Constraints in Quiz Clues
In crossword-style or compact quiz clues, a parenthetical number may specify the answer's length. When the format supports that reading, treat it as a constraint rather than dismissing it as a typo. Use the definition, wordplay, and any name or phrase completion together to find a concise answer that fits the stated length.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
