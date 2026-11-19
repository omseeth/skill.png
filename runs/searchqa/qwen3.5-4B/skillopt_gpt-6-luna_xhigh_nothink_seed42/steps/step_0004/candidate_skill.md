# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

## Resolve Visual References
When a clue refers to something “seen here” or otherwise points to a depiction, identify the depicted item rather than choosing a related item merely because it is prominent in retrieved text. Use the image when available; if it is unavailable, prefer the most specific evidence matching the clue and treat generic brand or category mentions as insufficient to identify the depicted item.

## Preserve Conventional Name Forms
For a person or other named entity, use the concise conventional form supported by the evidence. Preserve initials and abbreviations when they are the standard cited form; do not expand them unless the question requests a full name or the evidence requires it.
