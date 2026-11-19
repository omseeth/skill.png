# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Select and State the Requested Answer
Identify the exact answer slot the question or clue calls for (such as a person, group, attribute, or title), then give only the minimal answer that fills it. Treat clue-like fragments and declarative descriptions as implicit questions when they point to an omitted identity; give a true/false response only when the prompt explicitly asks for a truth judgment. Use relevant context to resolve the clue, even when the relationship is implicit, but do not invent an answer when context provides no supporting candidate. In trivia-style lists, clue text, category labels, and answer-length markers may run together with or without punctuation. A heading or label before a colon—or a trailing label or parenthetical number—may be metadata for the following clue, including when it appears immediately after the previous clue. Do not assign such metadata to the preceding clue or return it as the answer; solve the clue itself using its meaning and corroborating context. For fragmentary quotes, titles, nicknames, or familiar sayings and proverbs, infer the omitted word or phrase that completes the clue rather than echoing the supplied text. Return only the identifying portion requested: do not repeat title wording already supplied by the clue or append a generic class word when it is implicit, unless the full name is requested or needed to distinguish the answer. Use a name form that uniquely identifies the entity: give the complete identifying name when needed, but a surname or shorter form is appropriate when it is unambiguous. Do not add explanations or descriptive qualifiers unless requested.

For a prompt consisting only of a named work, slogan, or quoted expression, with or without a year, infer the intended identifying relation from context rather than defaulting to a definition or classification. When context clearly links the expression to a person or entity being identified, give that attribution; if the prompt asks for the expression's type or meaning, answer that instead. For a quoted catchphrase that also appears in a related work's title, prefer the familiar source work or speaker when context treats the phrase as a catchphrase; return the matching title only when the prompt or context clearly asks for that work.

## Match and Combine Contextual Evidence
Use distinctive details in the question—such as dates, roles, relationships, or defining descriptions—to find evidence that directly explains the full clue, rather than relying on a keyword-only match. If no single passage provides the answer, combine compatible details across relevant passages and corroborate the candidate when evidence is available. Ignore unrelated snippets.

In historical or archival clues, interpret time-relative terms such as “current” or “now” relative to the clue’s source or publication date when that frame is identifiable; do not automatically substitute the present-day officeholder for the person intended by an older clue.

## Check the Clue's Target, Scope, and Form

When choosing a name form, distinguish identity-bearing parts from generic type words supplied by the clue. Retain conventional initials that belong to a person's identifying name; omit a generic label such as an entity type or location type when the clue supplies that type and the shorter name identifies the answer unambiguously.

Treat grammatical relations as target cues: wording such as “this state's X” asks for the state, not X's divisions, members, or examples. Use context to identify the entity filling the possessive or relational slot.
Before choosing a candidate, use the clue's head nouns and pronouns to identify the kind of answer requested: a manner, place, role, institution, species, person, or other category. Return that target, not a consequence, associated entity, or example used to identify it. Match the clue's level of specificity—for example, a broad period rather than an exact year—and preserve the supported answer's conventional name, spelling, and grammatical number without adding modifiers that change its scope.

When a prompt consists only of a place name, do not simply repeat it. If context clearly identifies its jurisdiction or location, treat the prompt as an elliptical request for that place attribute and answer with the supported information. Do not infer a relation when context does not make one clear.

For a word clue that gives an etymology or source-language form alongside a modern definition or synonym, answer the modern headword described by the clue, not the etymon. Give the source word only when the prompt asks for the origin or original-language form.

When a clue uses a foreign-language word as a common noun, answer with its generic translated meaning at the same level of specificity. Do not narrow it to a contextually associated emblem, subtype, or proper name unless the clue explicitly asks for that identity.

When a prompt consists only of a list of character names or roles, use context to identify their shared work or program if that is the implied target. Return the work or program, not the performers, unless the prompt asks for the cast.

For a prompt consisting only of several people’s names, use context to identify the distinctive shared relation the list is meant to cue (for example, a common country or realm for rulers), rather than defaulting to a generic common role. Return the shared entity or attribute the clue calls for, not a succession or family chain; do not guess when context leaves multiple relations equally plausible or establishes none.

For compact definition or wordplay clues that give two linked readings, solve the whole clue for the conventional term satisfying both; do not merely restate one reading. Preserve an article when it is part of the expected expression.

For direct definition or trivia clues, treat the description as a fill-in target and return the conventional term or category called for by the whole clue, including after elliptical wording such as “one of these.” Match the requested grammatical number; when the answer is a singular count noun, include its conventional indefinite article. Do not answer with a named example when the clue asks for its class, or repeat an embedded synonym that serves only as a cue rather than the target.

Honor the measurement or attribute explicitly requested: for example, “smallest in size” asks about physical dimensions, not denomination or value. Do not substitute a related property for the stated one; use evidence about the requested property.

<!-- SLOW_UPDATE_START -->
When a clue explicitly asks for a kind of entity, make that the answer target and use the other names and roles only as evidence. For example, a clue asking for the film series associated with David Arquette’s Deputy Dewey role calls for Scream, not the performer, an incidental production, or an unrelated title. Return a performer only when the performer is the requested target.

For physical-size comparisons, compare physical dimensions; never infer size from denomination or value. The smallest U.S. coin currently minted by size is a dime, not a cent. When evidence concerns size, do not substitute evidence about monetary value.

Use the supported conventional name form: expand a nickname to the full given name when the evidence supports it, and omit optional middle initials or generational suffixes unless requested or needed to distinguish the person. Thus prefer Patricia Hearst, Lyndon Johnson, and Alan Shepard when those are the supported answer forms. Conversely, do not shorten a proper publication title by removing a word that belongs to it: when the journal is named Forbes magazine, return Forbes magazine rather than Forbes.

For a bare officeholder cue such as President Tran Duc Luong, check whether context makes the associated country the intended answer. If it does, return Vietnam rather than repeating the office title as President of Vietnam.
<!-- SLOW_UPDATE_END -->
