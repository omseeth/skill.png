# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Type the Unknown Before Answering

For an elliptical fragment without an explicit question, reconstruct the omitted relation from substantive context before filling the answer slot—for example, a date-and-place fragment may identify an event or person, while “name before X” asks for an earlier name of X. Resolve references only when the context supports them; do not infer a relation from a bare name or association alone.
For a clue-like prompt, identify the semantic type of the requested answer before retrieving it. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than treating the statement as a request to judge its truth. Reject candidates that are associated with the answer but are not the requested kind of thing.

For definition-style clues, match the requested part of speech, grammatical form, and number. Do not automatically reduce an activity or process noun to its base verb: the clue may call for a gerund or another inflected form. Conversely, a singular term or entity calls for a singular headword even when examples or supporting evidence are plural. Use the lemma only when the clue asks for it, and do not copy the number of nearby examples.

## Treat Clue Echoes as Search Cues
In noisy or search-style context, a passage may repeat the clue without supplying its answer. Treat that echo as a locator for relevant material, not as evidence for the missing answer; use substantive passages to identify the target and verify distinctive clue details. If the passage explicitly pairs the clue with an answer, it can serve as direct evidence.

## Reconstruct Flattened Clue Prompts
When a question is keyword-like, has lost spaces or punctuation, or comes from a clue archive, do not assume every adjacent word belongs to one phrase. Use the snippet and context to distinguish the clue from category labels, answer-length markers, and neighboring clue text. Identify the requested answer slot and answer that clue, rather than a related entity suggested by surrounding terms.

## Preserve Full Canonical Geographic Names
When a geographic proper name conventionally includes a generic component such as “Lake,” “Mount,” or “River,” retain that component in the answer. Do not shorten the name to its modifier unless the context or clue establishes the shortened form as the requested label.

## Verify the Clue's Exact Relation

When a clue includes a distinctive quotation, epithet, or slogan, use it as an identifying cue and determine the relation requested: the entity it describes, the person who said or coined it, or the referent it alludes to. Follow the clue's wording and contextual attribution rather than returning the quoted phrase itself.
For clues linking entities, identify the precise relation and the role of the missing answer before selecting a candidate—for example, a source versus what an instrument measures, or a city represented in an acronym versus the location of a related institution. A candidate's name or topic appearing nearby is not enough: prefer evidence that supports the same relation and the clue's distinguishing details.

## Use the Conventional Name at the Requested Scope
For proper names and titles, use the conventional form that identifies the requested entity. Do not substitute a nickname or later brand name for the historically relevant label, omit words from a work's title when the work is requested, or add optional middle names. Conversely, when the clue requests a generic class, do not embellish it with a narrower associated name.

## Test the Candidate in the Answer Slot
Before finalizing, substitute the candidate into the answer slot implied by the clue. It must match the requested semantic type and granularity—for example, a city rather than a conference held there, an acid rather than an alkaline cleaning agent, or a generic term rather than a named event or source-title phrase. For a term or class, use the answer at that scope; do not promote it to a fuller label unless the clue requests that name. Check that its grammatical form also fits the clue.

## Check for Shared Completions in Parallel Clues
When a clue presents alternatives or examples in parallel, check whether one word completes or classifies them both. The target may be that shared word rather than a full phrase inferred from just one part of the clue. Return the minimal shared completion unless the clue asks for a full phrase.

<!-- SLOW_UPDATE_START -->
Before submitting, test the candidate against the exact wording of the clue: return the requested entity or term itself, not a descriptive expansion of it. For example, when the clue identifies what has a given formula, answer with the named object rather than a phrase describing its area.

For people, use the full identity supported by substantive context; do not answer with only a surname merely because a clue echo abbreviates it. Preserve initials, titles, and other name elements when the clue or a reliable passage supplies them—do not strip them for stylistic normalization.

Check the candidate’s grammatical form by fitting it into the clue’s wording. A definition phrased with a bare verb does not mean the requested term is that verb: retain an activity or process form when the clue asks for that form, such as a term for following a course, rather than defaulting to its infinitive.
<!-- SLOW_UPDATE_END -->
