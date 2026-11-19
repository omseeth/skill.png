# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering

When a clue embeds or quotes a phrase, use the surrounding wording to determine what the quote is doing: it may describe a referent, supply text whose detail is requested, or itself be the wording to identify. Return the requested referent or detail rather than defaulting to the quote’s author, source, or title. Use context to resolve the clue and separate it from adjacent snippets that belong to another clue.
For a clue-like prompt, identify the semantic type and requested level of specificity before retrieving the answer. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

For named entities, use the complete conventional form needed to identify the target, preserving standard spelling and spacing rather than truncating a multiword name or substituting a related phrase. Match the clue’s scope: give a broad category when that is requested, not a narrower subtype merely mentioned in the clue; conversely, do not collapse a specifically requested subtype into its broader class.

## Preserve Direct Answer Forms
When context explicitly keys an answer or gives the canonical name or title of the identified target, preserve that form in the final answer. Do not expand a keyed short answer with extra name components, or reduce a multiword proper title to only one modifier.
