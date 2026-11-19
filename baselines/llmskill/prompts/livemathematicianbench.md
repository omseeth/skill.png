# Write a skill for LiveMathematicianBench

You are writing a skill: a compact Markdown document of guidance for a language model. The skill is appended to the model's system prompt as its own `## Skill` section, and the model then answers every task of a benchmark under that combined system prompt. The model's weights are frozen and the skill never changes once written. The same skill is used unchanged with several different models, with and without a separate reasoning phase.

## The benchmark

Each task is one research-level multiple-choice question about a theorem from a recent arXiv paper in mathematics, with five candidate statements labelled A to E. The user turn gives the question under `## Question` and the options under `## Choices`, with LaTeX kept as written. The model decides from the question and the options alone, without tools or references.

Exactly one option is correct: the strongest statement that holds. The other options are not correct.

## The system prompt the skill is appended to

The model receives this system prompt verbatim for every task. Your skill follows it.

<system_prompt>
{system_prompt}
</system_prompt>

## How an answer is read and scored

- The scorer reads the last `<answer>...</answer>` block. It must hold exactly one capital letter from A to E; parentheses, bold markers or a trailing period around the letter are tolerated. Anything else, such as two letters, a word, a lowercase letter or an empty tag, scores 0.
- There is no fallback: a response without a complete tag pair scores 0, however good the reasoning before it.
- The metric is accuracy.

## What a good skill for this benchmark covers

- Reading the question precisely: what exactly is asked, and which hypotheses, quantifiers and ranges it fixes.
- Comparing the options with each other rather than one at a time, to find the strongest statement that holds.
- Cheap tests in place of full proofs: small and extreme cases, known special cases, sign and scaling checks, and counterexamples that rule options out.
- The reasoning budget: this benchmark pushes models to reason at great length, and a response that runs past the output-token limit is cut off before its answer and scores 0. A good skill gets the model to a well-supported choice within bounded reasoning, and keeps it from re-deriving results or wavering between options once the evidence suffices.

The system prompt already asks the model to commit and avoid second-guessing. The skill should add procedure for doing so, not repeat it.

{examples}

## Requirements for the skill

- Write procedural, generalizable rules: how to read the task, how to decide, and how to check an answer before committing to it. Every rule must help on tasks you have not seen.
- Include no instance-specific content: no theorem statements, mathematical objects, question wording or answer letters from any particular task.
- Leave the answer format to the system prompt. Do not restate, rephrase, extend or override it, and do not contradict any of its rules; the skill adds to the system prompt and replaces no part of it.
- Address the model directly, in plain Markdown: short paragraphs, bullet lists, and headings of level three (`###`) or lower, since the skill already sits under a level-two heading. No code blocks, no preamble, no closing remarks.
- Keep it compact: at most about 1,500 words. Shorter is better when nothing is lost.

## Output

Return the skill and nothing else, wrapped in <skill> and </skill>.
