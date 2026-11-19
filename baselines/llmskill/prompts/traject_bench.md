# Write a skill for TRAJECT-Bench

You are writing a skill: a compact Markdown document of guidance for a language model. The skill is appended to the model's system prompt as its own `## Skill` section, and the model then answers every task of a benchmark under that combined system prompt. The model's weights are frozen and the skill never changes once written. The same skill is used unchanged with several different models, with and without a separate reasoning phase.

## The benchmark

Each task is a user query and a list of candidate tools. Everything the query asks for can be done with independent tool calls made at the same time, and only some of the listed tools are needed. The user turn lists the tools under `## Tools`, one block each, and then gives the query under `## Query`. A tool block gives the tool's name (`Provider: endpoint`), its provider with the provider's description, the API name, the tool's description, and its required and optional parameters with their types and descriptions. Nothing is executed and no results come back: the model writes every call in one answer.

## The system prompt the skill is appended to

The model receives this system prompt verbatim for every task. Your skill follows it.

<system_prompt>
{system_prompt}
</system_prompt>

## How an answer is read and scored

- The scorer reads the last `<answer>...</answer>` block, falling back to the last fenced block and then to the last JSON value in the output, and parses it as a JSON list of calls. The order of the calls, and of the parameters within a call, does not matter.
- A task counts as solved only when both of these hold: the set of tool names called equals the needed set exactly, with none missing and none extra; and every called tool's required parameters match the gold names and values exactly. One wrong or missing required value, or one missing or extra tool, scores the whole task 0. Optional parameters do not enter the score.
- Tool names are compared exactly as they are written in the tool list.
- Values are compared after light normalization: surrounding whitespace, a number written as a string or as a number, ISO dates with or without zero padding, spellings of true and false, and country codes inside comma-separated locations. Nothing else is folded: a synonym, an abbreviation, a converted unit or a reformatted value counts as wrong.

## What a good skill for this benchmark covers

- Decomposing the query into its separate requests and mapping each to exactly one tool, so that every request is covered and no tool is called without a request behind it.
- Telling similar tools apart: reading the provider, API name, description and parameter list to find the endpoint that does exactly what a request asks, not one that merely sounds related.
- Filling required parameters: which part of the query belongs in which parameter, taking values as the query states them, and never supplying identifiers, keys or placeholder values the query does not contain.
- Checking before answering: requests against calls, and each call against its tool's list of required parameters.

The system prompt already states the rules for tools and parameters. The skill should add procedure for applying them, not repeat them.

{examples}

## Requirements for the skill

- Write procedural, generalizable rules: how to read the task, how to decide, and how to check an answer before committing to it. Every rule must help on tasks you have not seen.
- Include no instance-specific content: no tool, provider or parameter names, no parameter values and no query wording from any particular task.
- Leave the answer format to the system prompt. Do not restate, rephrase, extend or override it, and do not contradict any of its rules; the skill adds to the system prompt and replaces no part of it.
- Address the model directly, in plain Markdown: short paragraphs, bullet lists, and headings of level three (`###`) or lower, since the skill already sits under a level-two heading. No code blocks, no preamble, no closing remarks.
- Keep it compact: at most about 1,500 words. Shorter is better when nothing is lost.

## Output

Return the skill and nothing else, wrapped in <skill> and </skill>.
