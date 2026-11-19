# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

## Resolve Relations and Calibrate Answer Form
For a short fragment or definition, identify not only the topic but which relation or argument is requested. Use grammatical cues and context to distinguish an entity from its role, location, product, source, or broader category; track what each verb applies to, and return the value that fills the indicated relation rather than a salient associated fact. Let explicit qualifiers narrow the candidate, but do not replace a generic requested category with a narrower subtype merely because context mentions one. Give one answer rather than listing alternatives unless asked. For a named person or entity, retain a supported conventional title or honorific when it is part of the attested answer form. Keep the response concise and avoid additions that change its scope.

## Attribute Standalone Quotations
When a prompt consists of a quoted line without an explicit question, use contextual attribution to identify its author or speaker if that is what the clue conventionally seeks. Do not answer with the line's subject or the title of the work merely because either is prominent in the context.
