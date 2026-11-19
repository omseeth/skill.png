# Tool Calling Skill

## Plan the calls
- Break the request into distinct requested outcomes and keep a checklist. Choose endpoints by their described operation, object, platform, and supported parameters—not just broad topic keywords. Cover every requested outcome, including the final clause; avoid unrelated calls and prefer a direct endpoint that fits the request.
- Combine compatible filters for the same result in one call when supported. Use separate calls for genuinely separate outcomes or values.

- For requests involving multiple items, include every item; if the endpoint accepts only one item per call, call it once for each item, and issue independent calls together. Map pagination constraints to available pagination parameters.

## Map parameters carefully
- Treat the tool definition as authoritative: use the exact tool name, put parameters in the correct required or optional fields, and supply every required value. Map explicit constraints—such as limits, sorting, locale, dates, and filters—to supported parameters with the right meaning and declared type.
- Preserve user-provided identifiers and search terms unless the schema requires another representation. Use documented formats and canonical codes where specified. Do not invent identifiers, unsupported values, arbitrary defaults, or empty placeholders; do not pretend an endpoint supports a constraint it cannot express. Include optional parameters when they convey a requested constraint.

## Final check
Confirm every requested outcome and supported constraint is covered, each call is relevant, and every argument is valid for its tool. Serialize the response in the required format and ordering; verify the JSON wrapper, keys, parameter structure, brackets, and separators.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
