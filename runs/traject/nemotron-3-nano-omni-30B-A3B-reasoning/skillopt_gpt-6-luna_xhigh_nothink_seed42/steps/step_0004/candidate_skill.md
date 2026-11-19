# Tool Calling Skill

## Match requested results to endpoints
- Choose an endpoint by the operation and artifact requested, not just by topic. For a named source, use a source-specific endpoint or selector when available; do not substitute a general search. Distinguish adjacent outputs such as financing offers, pricing, and warranty information.
- Track each distinct requested deliverable and map it to a call. Apply compatible filters for the same result in one call; do not add unrelated calls or leave a requested result uncovered.

## Map arguments to the endpoint schema
- Map each value to the exact documented parameter, respecting its type and identifier domain. Do not substitute a familiar-looking code for a different kind of identifier or treat a named section as a numeric page. Use canonical values only when supported by the request or tool documentation; do not fabricate opaque IDs.
- Attach each constraint to the call it filters, and include optional parameters only when requested or directly entailed by the task.
- For endpoints with hierarchical geographic filters, map each named place to its actual administrative level and supply any required parent levels. Do not place a city or locality in a country- or state-level parameter just because that parameter is required. Use canonical codes and administrative names when the mapping is unambiguous; do not fabricate a hierarchy when it is not.
- When an endpoint represents one user-provided value with multiple component parameters, split the value according to the documented meaning of each field; do not copy the complete value into a component field. For example, when SMTP inputs separate `username` and `domain`, use the mailbox local part for `username` if that is what the endpoint expects. If the catalog does not establish the component meanings, do not guess.

## Serialize to the active answer format
- Follow the current task's exact wrapper, keys, and parameter-entry schema. Emit one valid JSON array containing one call object per item; represent required and optional parameter collections as arrays of `{ "name": ..., "value": ... }` entries, including empty arrays where required.
- Do not duplicate keys, use parameter objects in place of arrays, or emit multiple top-level array fragments. Check that the completed answer parses and that parameter values use the expected JSON types.
