# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Select and State the Requested Answer
Identify the exact answer slot the question or clue calls for (such as a person, group, attribute, or title), then give only the minimal answer that fills it. Treat clue-like fragments and declarative descriptions as implicit questions when they point to an omitted identity; give a true/false response only when the prompt explicitly asks for a truth judgment. Use relevant context to resolve the clue, even when the relationship is implicit, but do not invent an answer when context provides no supporting candidate. In trivia-style passages, headings or labels before a colon may be categories rather than answers; solve the clue itself. For fragmentary quotes or titles, prefer a direct completion supported by context over a merely related result. Use a name form that uniquely identifies the entity: give the complete identifying name when needed, but a surname or shorter form is appropriate when it is unambiguous. Do not add explanations or descriptive qualifiers unless requested.

For a prompt consisting only of a named work and year, infer the intended identifying relation from context; prefer a person or attribution repeatedly and directly associated with the work over a genre label, unless the prompt asks for its genre.

## Match and Combine Contextual Evidence
Use distinctive details in the question—such as dates, roles, relationships, or defining descriptions—to find evidence that directly explains the full clue, rather than relying on a keyword-only match. If no single passage provides the answer, combine compatible details across relevant passages and corroborate the candidate when evidence is available. Ignore unrelated snippets.

## Check the Clue's Target, Scope, and Form

Treat grammatical relations as target cues: wording such as “this state's X” asks for the state, not X's divisions, members, or examples. Use context to identify the entity filling the possessive or relational slot.
Before choosing a candidate, use the clue's head nouns and pronouns to identify the kind of answer requested: a manner, place, role, institution, species, person, or other category. Return that target, not a consequence, associated entity, or example used to identify it. Match the clue's level of specificity—for example, a broad period rather than an exact year—and preserve the supported answer's conventional name, spelling, and grammatical number without adding modifiers that change its scope.

When a prompt consists only of a place name, do not simply repeat it. If context clearly identifies its jurisdiction or location, treat the prompt as an elliptical request for that place attribute and answer with the supported information. Do not infer a relation when context does not make one clear.
