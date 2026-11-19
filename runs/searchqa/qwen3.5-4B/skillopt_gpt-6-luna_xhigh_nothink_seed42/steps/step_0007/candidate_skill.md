# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Identify the Answer Type and Slot Before Answering
For a clue-like prompt, identify both the semantic type requested and the exact slot to fill (for example, a person, place, season, or expansion of an abbreviation). In a full sentence or fragment, distinguish the clause expressing the requested relation from rhetorical lead-ins, modifiers, and nearby descriptive facts; do not return one of those merely because it is salient. Preserve the head noun in phrases such as “this phenomenon”: answer with a phenomenon, not an associated personification. In “a person like X” constructions, treat X as an example and give the relevant shared role or category. If a declarative clue describes an entity using a pronoun, resolve the pronoun’s referent rather than judging the statement’s truth. Test a candidate by substituting it into the requested slot, and reject candidates that are merely associated with the answer or do not fit the requested relation.

When a clue gives a number, proportion, date, or range, compare candidate evidence to that qualifier directly. Check which entity or subgroup the figure describes, and do not transfer a nearby statistic from a related but different candidate.

## Use the Best-Supported Answer Form
Use the lexical form that directly fills the requested slot, rather than a related inflection or part of speech. Let the question’s wording determine number and form; do not automatically normalize them. When the question does not require a particular form, prefer the concise conventional form supported by context, such as a standard singular lemma or a proper name without an unnecessary generic suffix. Give the shortest source-supported answer that identifies the target, and avoid optional refinements or secondary examples unless needed to disambiguate or explicitly requested.
