# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering

Treat colorful, approximate, or potentially conflicting details as clues, not necessarily as the requested answer. If the context independently identifies the entity asked for, answer that entity rather than derailing into a fact-check of an incidental detail. If the requested fact itself depends on disputed evidence, limit the answer to what the context supports.
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

## Parse the Clue's Requested Relation
For a declarative or fragmentary clue, determine which entity the clue asks for by parsing its relationships, not just its topic. A phrase such as “in this series” asks for the series, not the character or role mentioned; a clue about a group may point to the work that names it, not its size. For a bare list of names, use context to identify the shared named entity or source rather than labeling the list members. Prefer evidence that matches the whole clue, and disregard incidental neighboring material.

## Match the Requested Naming Level
Use the conventional answer form at the specificity the clue requests. Retain meaningful parts of a proper name, such as a location qualifier or ordinal, when they identify the target; do not expand an established short title with an optional subtitle. For ordinary nouns, answer at the requested category level rather than substituting a narrower subtype. When context provides a keyed answer form, prefer it.
