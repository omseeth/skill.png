# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

## Resolve the Clue's Missing Slot
Translate the whole clue into a relation with an unknown slot before choosing a candidate: for example, identify which co-performer is requested, what name others used for someone, what thing a former label refers to, or what work a list of names points to. Do not substitute a prominent associated person, object, or category for the requested referent. Check that the candidate fits all explicit clue details and the relevant context, including its role and relation; when the context supports a narrower subtype than the clue's generic noun, give that narrower answer. For fragments or lists, infer the linked entity the clue points to rather than merely naming the category of the listed items.

## Prefer Direct Clue Matches in Context
When a context passage repeats or closely paraphrases the clue and directly supplies an answer, use that evidence rather than reconstructing an answer from nearby topics or unsupported outside associations. Do not invent an explanation to justify a conflicting candidate.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
