# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

For definition-style clues, use the clue's grammatical role to select the headword: for example, a clue asking for “the cry” calls for a noun, while “to emit” or “to show” calls for a verb. Prefer the standard lemma over a related inflection or synonym.

## Treat Clue Echoes as Search Cues
In noisy or search-style context, a passage may repeat the clue without supplying its answer. Treat that echo as a locator for relevant material, not as evidence for the missing answer; use substantive passages to identify the target and verify distinctive clue details. If the passage explicitly pairs the clue with an answer, it can serve as direct evidence.

## Preserve Full Canonical Geographic Names
When a geographic proper name conventionally includes a generic component such as “Lake,” “Mount,” or “River,” retain that component in the answer. Do not shorten the name to its modifier unless the context or clue establishes the shortened form as the requested label.

## Verify the Clue's Exact Relation
For clues linking entities, identify the precise relation and the role of the missing answer before selecting a candidate—for example, a source versus what an instrument measures, or a city represented in an acronym versus the location of a related institution. A candidate's name or topic appearing nearby is not enough: prefer evidence that supports the same relation and the clue's distinguishing details.

## Use the Conventional Name at the Requested Scope
For proper names and titles, use the conventional label at the scope the clue asks for; do not assume the longest formal name is required. An established short title or company label is sufficient when it uniquely identifies the target and matches the requested scope, so do not add a subtitle or corporate suffix merely because a source uses one. Conversely, when the requested answer is a compound named entity or the clue calls for its full name, give its complete conventional name rather than only the missing word in a cloze. Preserve historically relevant labels rather than substituting a nickname or later brand name; retain meaningful distinguishing components when the clue requires them, and do not add optional middle names. When the clue requests a generic class, do not embellish it with a narrower associated name.

## Match the Supported Answer Form
For short-answer tasks, return the requested target alone; do not wrap a country, value, or label in an unrequested title or role phrase. Preserve the conventional, context-supported number of common nouns rather than automatically converting them to singular—some foods, species, and categories are conventionally expressed in the plural.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
