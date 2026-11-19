# Write a skill for SearchQA

You are writing a skill: a compact Markdown document of guidance for a language model. The skill is appended to the model's system prompt as its own `## Skill` section, and the model then answers every task of a benchmark under that combined system prompt. The model's weights are frozen and the skill never changes once written. The same skill is used unchanged with several different models, with and without a separate reasoning phase.

## The benchmark

Each task is a Jeopardy! clue together with web search snippets retrieved for it. The user turn gives the snippets under `## Context`, in their raw form: each document starts with a `[DOC]` marker, with `[TLE]` before its title and `[PAR]` before its text. The clue follows under `## Question`. The model reads the snippets and answers in one turn, without tools.

## The system prompt the skill is appended to

The model receives this system prompt verbatim for every task. Your skill follows it.

<system_prompt>
{system_prompt}
</system_prompt>

## How an answer is read and scored

- The scorer reads the last `<answer>...</answer>` block anywhere in the output. When the output has no complete tag pair, it falls back to the last non-empty line.
- The extracted answer is normalized (lowercased, punctuation removed, the articles "a", "an" and "the" removed, whitespace collapsed) and compared for exact equality with the gold answer, normalized the same way.
- Exact match is all or nothing. The right entity with anything extra, such as a descriptor, a title, a second candidate or an explanation, or with part of it missing, scores 0.
- A response cut off at the output-token limit is scored on the last answer tag it contains, even one it only quoted.

## What a good skill for this benchmark covers

- Reading the clue: clues state a fact about the answer rather than ask a question, often with "this" or "these" standing in for it, and sometimes with wordplay. Work out what kind of thing the answer is and every constraint the clue places on it.
- Finding and choosing candidates in the snippets, which can be noisy, partly off-topic and contradictory: prefer the candidate that satisfies every constraint of the clue and that independent snippets agree on.
- Choosing the surface form, which decides the score: the standard name as the snippets write it, without descriptors or hedging, and with care where a name has a short and a long form.
- Finishing: reaching a decision within bounded reasoning, because an answer that is never written scores 0.

The system prompt already states the answer format and asks for concise answers. The skill should add procedure for doing the task well, not repeat those rules.

{examples}

## Requirements for the skill

- Write procedural, generalizable rules: how to read the task, how to decide, and how to check an answer before committing to it. Every rule must help on tasks you have not seen.
- Include no instance-specific content: no entities, clue wording, snippet text or answers from any particular task.
- Leave the answer format to the system prompt. Do not restate, rephrase, extend or override it, and do not contradict any of its rules; the skill adds to the system prompt and replaces no part of it.
- Address the model directly, in plain Markdown: short paragraphs, bullet lists, and headings of level three (`###`) or lower, since the skill already sits under a level-two heading. No code blocks, no preamble, no closing remarks.
- Keep it compact: at most about 1,500 words. Shorter is better when nothing is lost.

## Output

Return the skill and nothing else, wrapped in <skill> and </skill>.
