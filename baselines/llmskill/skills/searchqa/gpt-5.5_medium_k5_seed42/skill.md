### SearchQA Procedure

### 1. Parse the clue first
- Treat the clue as a statement whose missing subject/object is the answer.
- Identify the expected answer type: person, place, organization, work, event, object, term, phrase, date, etc.
- List all constraints in the clue: actions, relationships, dates, locations, titles, nicknames, quantities, quoted phrases, category hints, and any wordplay.
- Pay special attention to pronouns such as “this,” “these,” “he,” “she,” or “it”: they signal the thing to name.

### 2. Use snippets as evidence, not as a single source of truth
- Search the context for candidates matching the expected type.
- Prefer candidates that satisfy every clue constraint, not merely one prominent keyword.
- Give more weight to snippets that directly connect the candidate to the distinctive clue facts.
- Independent agreement across multiple snippets is strong evidence.
- Ignore snippets that are off-topic, only share a keyword, or contradict stronger/direct evidence.

### 3. Handle Jeopardy-style clue patterns
- If the clue gives a definition or description, answer with the named term/entity described.
- If the clue gives a nickname, alias, or “also known as,” answer with the underlying thing unless the clue asks for the nickname itself.
- If the clue contains a quotation or altered phrase, infer the intended original phrase or referenced expression when the snippets indicate one.
- If the clue asks for a country, city, company, author, work, etc., do not answer with a related broader/narrower entity unless it exactly fits the type.
- For dates or events, ensure the candidate is tied to the exact time and action in the clue.

### 4. Resolve competing candidates
- Check each plausible candidate against all constraints; eliminate any that fail one clear constraint.
- If snippets mention both a long formal name and a common short name, choose the form most likely accepted as the standard answer and supported by the clue.
- If a title and creator/performer/person are both present, answer the one matching the clue’s grammatical role and expected type.
- If several related entities appear, prefer the one that is the direct subject of the clue fact, not an associated example or location.

### 5. Choose the exact surface form
- Use the standard name as written in reliable/direct snippets.
- Do not include descriptions, appositives, explanations, parentheticals, dates, or uncertainty.
- Do not give multiple alternatives.
- Avoid adding articles unless they are normally part of the name or title.
- Preserve necessary words in multiword names and titles; do not shorten to a partial name if the full standard form is needed.
- Conversely, do not use an overlong formal name when the snippets and clue point to the common accepted name.

### 6. Final check before committing
- Does the answer have the right entity type?
- Does it satisfy every distinctive clue constraint?
- Is it the direct answer, not merely related background?
- Is the wording clean: one answer, no extra qualifiers, no hedging?
- If uncertain, choose the candidate with the strongest direct snippet support and still provide a final answer.
