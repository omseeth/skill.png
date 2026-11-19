# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

For definition-style clues, match the requested part of speech, grammatical form, and number. Do not automatically reduce an activity or process noun to its base verb: the clue may call for a gerund or another inflected form. Conversely, a singular term or entity calls for a singular headword even when examples or supporting evidence are plural. Use the lemma only when the clue asks for it, and do not copy the number of nearby examples.

## Treat Clue Echoes as Search Cues
In noisy or search-style context, a passage may repeat the clue without supplying its answer. Treat that echo as a locator for relevant material, not as evidence for the missing answer; use substantive passages to identify the target and verify distinctive clue details. If the passage explicitly pairs the clue with an answer, it can serve as direct evidence.

## Preserve Full Canonical Geographic Names
When answering with a geographic proper name, retain intrinsic generic components conventionally belonging to it, such as “Lake,” “Mount,” or “River.” Do not confuse these with administrative qualifiers: honor the geographic level requested by the clue. If it asks for a country, return the country rather than a state or region; if it asks for the jurisdiction of a named place, return that jurisdiction alone, not the place plus its jurisdiction. Include a jurisdiction only when the clue requests it or it is part of the requested name.

## Verify the Clue's Exact Relation

When a clue includes a distinctive quotation, epithet, or slogan, use it as an identifying cue and determine the relation requested: the entity it describes, the person who said or coined it, or the referent it alludes to. Follow the clue's wording and contextual attribution rather than returning the quoted phrase itself.
For clues linking entities, identify the precise relation and the role of the missing answer before selecting a candidate—for example, a source versus what an instrument measures, or a city represented in an acronym versus the location of a related institution. A candidate's name or topic appearing nearby is not enough: prefer evidence that supports the same relation and the clue's distinguishing details.

## Use the Conventional Name at the Requested Scope
For proper names and titles, use the conventional form that identifies the requested entity at the clue's scope and in its historical or contextual frame. When a clue asks for a company or brand by its familiar public name, do not expand that name into a longer corporate name or append a product-category word just because a source uses it. Do not substitute a nickname or later brand name for a historically relevant label, omit words intrinsic to a requested work's title, or add optional middle names. Conversely, when the clue requests a generic class, do not embellish it with a narrower associated name.

## Check for Shared Completions in Parallel Clues
When a clue presents alternatives or examples in parallel, check whether one word completes or classifies them both. The target may be that shared word rather than a full phrase inferred from just one part of the clue. Return the minimal shared completion unless the clue asks for a full phrase.

<!-- SLOW_UPDATE_START -->
Before submitting, test the candidate against the exact wording of the clue: return the requested entity or term itself, not a descriptive expansion of it. For example, when the clue identifies what has a given formula, answer with the named object rather than a phrase describing its area.

For people, use the full identity supported by substantive context; do not answer with only a surname merely because a clue echo abbreviates it. Preserve initials, titles, and other name elements when the clue or a reliable passage supplies them—do not strip them for stylistic normalization.

Check the candidate’s grammatical form by fitting it into the clue’s wording. A definition phrased with a bare verb does not mean the requested term is that verb: retain an activity or process form when the clue asks for that form, such as a term for following a course, rather than defaulting to its infinitive.
<!-- SLOW_UPDATE_END -->
