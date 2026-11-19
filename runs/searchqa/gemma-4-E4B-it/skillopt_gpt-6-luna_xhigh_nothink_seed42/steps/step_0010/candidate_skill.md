# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Select and State the Requested Answer
Identify the exact answer slot the question or clue calls for (such as a person, group, attribute, or title), then give only the minimal answer that fills it. Treat clue-like fragments and declarative descriptions as implicit questions when they point to an omitted identity; give a true/false response only when the prompt explicitly asks for a truth judgment. Use relevant context to resolve the clue, even when the relationship is implicit, but do not invent an answer when context provides no supporting candidate. In trivia-style lists, clue text, category labels, and answer-length markers may run together with or without punctuation. A heading or label before a colon—or a trailing label or parenthetical number—may be metadata for the following clue, including when it appears immediately after the previous clue. Do not assign such metadata to the preceding clue or return it as the answer; solve the clue itself using its meaning and corroborating context. For fragmentary quotes, nicknames, or familiar sayings and proverbs, infer the omitted word or phrase that completes the clue rather than echoing the supplied text. When a clue asks for a named work but supplies only a fragment of its title, return the complete conventional title, including words already supplied by the clue. Return only the omitted word or words when the prompt explicitly asks to complete a blank or fragment. Return only the identifying portion requested: do not repeat title wording already supplied by the clue or append a generic class word when it is implicit, unless the full name is requested or needed to distinguish the answer. Use a name form that uniquely identifies the entity: give the complete identifying name when needed, but a surname or shorter form is appropriate when it is unambiguous. Do not add explanations or descriptive qualifiers unless requested.

For a prompt consisting only of a named work, slogan, or quoted expression, with or without a year, infer the intended identifying relation from context rather than defaulting to a definition or classification. When context clearly links the expression to a person or entity being identified, give that attribution; if the prompt asks for the expression's type or meaning, answer that instead.

## Match and Combine Contextual Evidence
Use distinctive details in the question—such as dates, roles, relationships, or defining descriptions—to find evidence that directly explains the full clue, rather than relying on a keyword-only match. If no single passage provides the answer, combine compatible details across relevant passages and corroborate the candidate when evidence is available. Ignore unrelated snippets.

### Enforce Enumerated-Set Completion
When a clue names several members of a set and asks for a missing or remaining member, treat the list as a hard exclusion constraint: choose an omitted member that also fits the descriptive clue. Reject any candidate already named, even if context associates it with a salient keyword; resolve the whole completion rather than one phrase in isolation.

## Check the Clue's Target, Scope, and Form

When choosing a name form, distinguish identity-bearing parts from generic type words supplied by the clue. Retain conventional initials that belong to a person's identifying name; omit a generic label such as an entity type or location type when the clue supplies that type and the shorter name identifies the answer unambiguously.

Treat grammatical relations as target cues: wording such as “this state's X” asks for the state, not X's divisions, members, or examples. Use context to identify the entity filling the possessive or relational slot.

For clues about advancing through an ordered series of ranks, count each successive rank after the stated starting rank to find the endpoint. Then match the requested answer slot: a clue asking who someone becomes or for “one of these” calls for the person or member designation, not merely the rank label.
Before choosing a candidate, use the clue's head nouns and pronouns to identify the kind of answer requested: a manner, place, role, institution, species, person, or other category. Return that target, not a consequence, associated entity, or example used to identify it. Match the clue's level of specificity—for example, a broad period rather than an exact year—and preserve the supported answer's conventional name, spelling, and grammatical number without adding modifiers that change its scope.

When a prompt consists only of a place name, do not simply repeat it. Treat it as an elliptical request for the place’s country or administrative location only when context supports that relation. For a city, give the country or jurisdiction stated in context, not a component of its formal style or name, unless the prompt asks for its formal name or status. Do not infer a relation when context does not make one clear.

For a word clue that gives an etymology or source-language form alongside a modern definition or synonym, answer the modern headword described by the clue, not the etymon. Give the source word only when the prompt asks for the origin or original-language form.

When a prompt consists only of a list of character names or roles, use context to identify their shared work or program if that is the implied target. Return the work or program, not the performers, unless the prompt asks for the cast.

For compact definition or wordplay clues that give two linked readings, solve the whole clue for the conventional term satisfying both; do not merely restate one reading. Preserve an article when it is part of the expected expression.

Honor the measurement or attribute explicitly requested: for example, “smallest in size” asks about physical dimensions, not denomination or value. Do not substitute a related property for the stated one; use evidence about the requested property.
