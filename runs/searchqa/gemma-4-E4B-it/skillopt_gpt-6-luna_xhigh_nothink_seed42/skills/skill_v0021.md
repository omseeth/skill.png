# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Select and State the Requested Answer
Identify the exact answer slot the question or clue calls for (such as a person, group, attribute, or title), then give only the minimal answer that fills it. Treat clue-like fragments and declarative descriptions as implicit questions when they point to an omitted identity; give a true/false response only when the prompt explicitly asks for a truth judgment. Use relevant context to resolve the clue, even when the relationship is implicit, but do not invent an answer when context provides no supporting candidate. In trivia-style lists, clue text, category labels, and answer-length markers may run together with or without punctuation. A heading or label before a colon—or a trailing label or parenthetical number—may be metadata for the following clue, including when it appears immediately after the previous clue. Do not assign such metadata to the preceding clue or return it as the answer; solve the clue itself using its meaning and corroborating context. For fragmentary quotes, titles, nicknames, or familiar sayings and proverbs, infer the omitted word or phrase that completes the clue rather than echoing the supplied text. Return only the identifying portion requested: do not repeat title wording already supplied by the clue or append a generic class word when it is implicit, unless the full name is requested or needed to distinguish the answer. Use a name form that uniquely identifies the entity: give the complete identifying name when needed, but a surname or shorter form is appropriate when it is unambiguous. Do not add explanations or descriptive qualifiers unless requested.

For a prompt consisting only of a named work, slogan, or quoted expression, with or without a year, infer the intended identifying relation from context rather than defaulting to a definition or classification. When context clearly links the expression to a person or entity being identified, give that attribution; if the prompt asks for the expression's type or meaning, answer that instead.

## Match and Combine Contextual Evidence
Use distinctive details in the question—such as dates, roles, relationships, or defining descriptions—to find evidence that directly explains the full clue, rather than relying on a keyword-only match. If no single passage provides the answer, combine compatible details across relevant passages and corroborate the candidate when evidence is available. Ignore unrelated snippets.

## Check the Clue's Target, Scope, and Form

When choosing a name form, distinguish identity-bearing parts from generic type words supplied by the clue. Retain conventional initials that belong to a person's identifying name; omit a generic label such as an entity type or location type when the clue supplies that type and the shorter name identifies the answer unambiguously.

Treat grammatical relations as target cues: wording such as “this state's X” asks for the state, not X's divisions, members, or examples. Use context to identify the entity filling the possessive or relational slot.
Before choosing a candidate, use the clue's head nouns and pronouns to identify the kind of answer requested: a manner, place, role, institution, species, person, or other category. Return that target, not a consequence, associated entity, or example used to identify it. Match the clue's level of specificity—for example, a broad period rather than an exact year—and preserve the supported answer's conventional name, spelling, and grammatical number without adding modifiers that change its scope.

When a prompt consists only of a place name, do not simply repeat it. If context clearly identifies its jurisdiction or location, treat the prompt as an elliptical request for that place attribute and answer with the supported information. Do not infer a relation when context does not make one clear.

For a word clue that gives an etymology or source-language form alongside a modern definition or synonym, answer the modern headword described by the clue, not the etymon. Give the source word only when the prompt asks for the origin or original-language form.

When a prompt consists only of a list of character names or roles, use context to identify their shared work or program if that is the implied target. Return the work or program, not the performers, unless the prompt asks for the cast.

For compact definition or wordplay clues that give two linked readings, solve the whole clue for the conventional term satisfying both; do not merely restate one reading. Preserve an article when it is part of the expected expression.

Honor the measurement or attribute explicitly requested: for example, “smallest in size” asks about physical dimensions, not denomination or value. Do not substitute a related property for the stated one; use evidence about the requested property.

<!-- SLOW_UPDATE_START -->
For a clue-like biographical or event statement with an unnamed he or she as its subject, identify that person as the answer unless the prompt explicitly asks for the event, work, or other detail. Do not answer with a named book or event merely because it appears in the clue. When a clue identifies someone by saying they played one or more roles, return the performer; use the productions as evidence, not as the answer. When an unnamed it or its is characterized by a defining property or formula, return the entity that has that property, not the formula itself.

When a clue asks what word preceded or completed multiple cited titles, find the exact wording that fits all the titles and return only that missing wording; do not substitute a category label or unrelated fragment. For definition clues that include an etymology, match the exact modern word form supported by the definition and clue wording, including an inflected form when applicable; do not replace it with a base form or the source word. Preserve conventional honorifics such as Sir when the person is identified in that styled form by the clue or supporting evidence.
<!-- SLOW_UPDATE_END -->
