# Tool Calling Skill

## Request coverage and endpoint choice
- Break the request into a checklist of distinct requested outcomes, actions, and constraints before selecting tools. Account for every independent subtask, including side requests, and preserve separate requested results. Make separate calls when distinct results require them and the endpoint cannot handle them together; do not add redundant, merely related, or unrequested calls.
- Use the available tool catalog as the source of truth. For each action, choose the listed endpoint whose documented function most specifically fits it. Do not substitute a merely related endpoint or invent an unavailable one. Use the exact fully qualified tool name shown in the catalog (`Provider: endpoint`), not an alias or shortened name.

## Argument validation
- Map each requested identifier, filter, and constraint to the endpoint's exact parameter name. Respect required and optional classifications, documented types, accepted enum or code formats, and supported value ranges. Carry explicit constraints, such as a requested result count, into the corresponding supported parameter rather than silently relying on a default.
- Include explicitly requested optional parameters, but omit unrelated ones. Every included argument must have a concrete, supported value; do not invent missing identifiers, dates, credentials, choices, or placeholders. Use a documented default only when it is needed to fulfill the request.

## Final validation
- Compare the calls against the request checklist: every requested outcome must be covered, with no extraneous calls.
- Validate the output against the calling interface: use the required wrapper and keys, valid JSON syntax, exact tool names, and the required parameter arrays. Where parameter entries are required, use `{ "name": ..., "value": ... }` objects rather than bare names, and check for duplicate keys. Sort calls by tool name when required by the interface.

## Efficient execution planning
- Combine compatible constraints for the same result in one call. If an endpoint handles one item at a time, make a separate call for each requested entity and carry over shared parameters. When required inputs are already provided, avoid unnecessary search or lookup calls. Make independent calls together when the interface supports it.
