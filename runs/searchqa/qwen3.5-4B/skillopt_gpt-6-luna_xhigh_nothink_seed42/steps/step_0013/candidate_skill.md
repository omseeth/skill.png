# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing. When the context reproduces the same clue in a structured question-and-answer source and explicitly pairs it with an answer, treat that pairing as direct evidence, while confirming the answer matches the requested type.

## Resolve the Clue's Relation and Scope
Identify which entity the clue asks for and the relation connecting it to the surrounding facts. For fragmentary or elliptical clues, infer only a relation supported by the wording and context; prefer evidence that states the relation directly over a nearby but different fact. Do not substitute a related entity at another level, such as a structure for its site or a region for its country. Track the direction of transformations: if the clue asks for a source or original title, do not return a later altered or derivative title.

## Give One Canonical Answer at the Right Granularity
When a short-answer clue calls for one term or label, choose the best-supported conventional answer and give only that answer; do not list alternatives unless asked. Use the shortest conventional wording that uniquely identifies the requested target. Do not repeat name components already supplied by the clue, but retain components needed to distinguish the entity, such as a numeral in a title. For terms and count nouns, use the canonical wording and number supported by context, and do not add an unnecessary modifier.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
