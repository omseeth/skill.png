# Mathematical Multiple-Choice Skill

## Exact option discrimination
When answer choices are near-duplicates, do not select by familiarity, detail, or how impressive the terminology sounds. Treat every option as a precise formal statement and compare the exact differences before choosing. Check especially:
- constants, coefficients, exponents, denominators, and endpoint inclusion such as `<` vs. `≤`;
- nested scales or formulas that look similar but are not identical;
- hypotheses such as finite, generic, injective, compatible, invariant, or point-separating assumptions;
- global vs. local statements, one-sided vs. two-sided estimates, and open vs. closed intervals;
- whether the option answers the requested quantity directly or gives only a related reformulation;
- order of parameters, arguments, blocks, or permutations inside patterned notation.

The correct option is the one matching the theorem, definition, or prompt exactly, not the one that is most familiar or most elaborate.

## Strength/provability discipline

For “strongest statement” questions, compare strength within the theorem's stated scope. Do not broaden the quantified domain—for example from a special class to all objects, from 2-colorings to all finite colorings, or from prime fields to all finite fields—merely because the generalization seems plausible. A broader-domain option is correct only when the cited result explicitly includes that broader scope.

When a theorem first gives a conclusion only after applying a map, do not lift it back to the original object unless the hypotheses provide injectivity, point-separation, or another exact lifting condition. “The image has property P” is weaker than “the original point has property P”; a condition such as “the linear system separates points” may be essential rather than decorative.

In classification or rigidity questions, beware tautological necessary conditions such as “isomorphic objects have the same invariant/spectrum.” Such statements may be true but too weak when a special theorem asserts a converse implication under narrow hypotheses. Do not use generic counterexamples from a larger category to reject a rigidity statement under the stated exceptional hypotheses.

When options differ between “unique object” and “unique up to scaling, normalization, orbit, or projective equivalence,” treat these as incompatible. A ray or orbit of minimizers is not the same as a single raw minimizer; check whether the problem is formulated on raw objects or on normalized/projectivized equivalence classes.
For “strongest statement” questions, strongest means strongest among statements actually justified by the stated hypotheses. A stronger-looking option is correct only if every extra clause is supported; a weaker-looking option is wrong if the hypotheses really imply the stronger one.

Do not upgrade to canonical bases, special constructions, sharper constants, endpoint regularity, boundedness, uniformity, finite-time/global behavior, two-sided estimates, multiplicative compatibility, or extra modes unless the theorem explicitly gives them. Prefer the minimal exact theorem statement over an over-specific strengthening when the extra property is not forced.

Treat every added refinement as a separate theorem. An option becomes false if it adds an unsupported two-sided estimate, lower bound, prefactor, a priori norm estimate, Taylor/asymptotic expansion, exact asymptotic equivalence `~`, sharper numerical bound, positivity/sign conclusion, stability/bounded-distance clause, uniform-in-time constant when only finite-interval bounds are given, orbit-classification when only invariance is known, or regularity of all weak solutions when uniqueness is only stated within a special class. Conversely, when the theorem explicitly guarantees an explicit witness, metric, or construction, that constructive statement is stronger than a merely abstract existence or metrizability claim.

## Quantifiers, hypotheses, and parameter dependence
Track quantifiers literally. Distinguish:
- `for every` vs. `for some`;
- a single object working for all parameters vs. a parameter-indexed family;
- constants independent of a parameter vs. constants or ellipticity bounds that grow with it;

- in asymptotic notation with a fixed parameter, whether the hidden `O` constant may depend on that fixed parameter or is asserted to be absolute or uniform over all parameter values;
- a property of every object vs. only objects satisfying an extra condition;
- necessary, sufficient, and biconditional statements;
- “there exists a unique solution in class X” vs. “every solution has property X.”

Do not invent caveats from general intuition. If the problem states smoothness, compatibility, dimension restrictions, positivity, or other hypotheses, first assume they are included to make the intended theorem true. Do not add unmentioned smallness assumptions, blow-up exceptions, loss of regularity, bad parameter ranges, torsion/orientation exceptions, or hidden genericity requirements unless forced by the theorem or by the choices.

## Constants in local estimates
For local Hölder/Lipschitz or similar estimates, distinguish fixed global constants from constants that can be absorbed by shrinking the neighborhood. If a theorem gives a local bound like `|f(x)-f(y)| ≤ C |x-y|^β` and the question allows choosing a sufficiently small neighborhood, a normalized coefficient such as `1` may be the intended local statement. Always check whether the neighborhood is existential before rejecting a normalized constant.

## Definitions, notation, and off-by-one counts
Before applying a formula, restate exactly what the prompt defines and what is counted or excluded: origins, endpoints, boundary components, anchors, total counts vs. counts away from an excluded item, and coordinate conventions. Do not replace `a` by `a+1`, or vice versa, unless the definition explicitly requires it.

Use specialized notation exactly as given. For block notation, ordered tuples, summation indices, or paper-specific definitions, compare the resulting argument order and roles literally rather than substituting a familiar convention.

## Equivalence and specialized notions
For equivalence or complexity questions, do not reject an option merely because it looks asymptotically stronger or weaker under ordinary implication intuition. The theorem may assert equivalence under the given structural hypotheses, so compare the exact equivalence claimed by the choices.

Unfamiliar named notions may not obey the standard analogue. For specialized variants or paper-specific definitions, do not substitute the nearest classical theorem if the wording of the problem and choices indicate a different first-order condition, extremal property, density, or global inequality.

## Additional theorem-grounded distinctions

Do not import threshold phenomena from analogous theories. When choices propose a universal statement versus a familiar cutoff, first identify whether the theorem actually asserts a sharp range, a uniform comparison, or uniform failure. A known-looking phase transition is not a reason to reject an exact universal option unless the stated result includes that transition.

For threshold or phase-regime choices, compare the regime split literally. Do not assume separate subcritical, critical, and supercritical behavior if the theorem groups all parameters on one side, such as `u ≥ u_c`, into the same case because of conditioning, truncation, or a boundary singularity.

For blow-up and divergence criteria, remember that `limsup = -∞` is an eventual divergence condition while `liminf = -∞` may be only subsequential divergence unless the theorem proves equivalence. Match the theorem's stated limit notion rather than substituting the more familiar one.

When the prompt asks what holds for every object or function, distinguish an unconditional parametrization statement from a conditional invariance statement. “Every object can be expressed as a function of parameters” is not invalid merely because it depends on the parameter; invariance or constancy is an extra condition and may be a weaker, conditional answer.

For complex probabilistic, geometric, or topological events, do not replace the event by a simpler necessary hitting or return calculation. Match nested scales such as `log`, `log log`, and powers thereof to the stated event, especially when the event involves disjoint subpaths, separation, or other global geometry.
When comparing convergence or limiting-behavior options, keep the exact limit notion: `limsup` vs. `liminf`, subsequential vs. eventual convergence, and local/restricted convergence vs. global/all-index convergence can determine the answer.

When the prompt asks for a full result, prefer the option that gives the complete regime or case description. A shorter statement may be true but incomplete if it omits a phase, exceptional case, positivity/full-measure/probability clause, or equality conclusion required by the theorem.

For quantitative results, distinguish the theorem's actual precision level. A Big-O upper bound is not an asymptotic formula, a one-sided existence bound is not a sharp two-sided characterization, and a membership/regularity conclusion is not automatically an estimate with specified dependence on data. For numerical “strongest estimate” questions, every larger lower bound or sharper constant is a stronger claim that must be justified by the stated result, not by memory of related tables or constructions.

Separate decision claims from constructive claims. A theorem or algorithm that decides whether conditions hold is weaker than one that constructs explicit witnesses; choose the constructive version only when explicit production is actually guaranteed.

## Witnesses, simultaneous choices, and bundled clauses
Parse the order of witnesses and targets literally. An existential witness placed before a universal target may require one fixed object to work for all targets; if it appears after the target, the witness may depend on that target. Likewise, when a prompt fixes a parameter, do not upgrade the result to a simultaneous family for all parameter values or to choices compatible across parameters unless that uniform construction is explicitly stated.

Match promised outputs exactly. An algorithm or theorem that produces the requested object is not automatically one that also gives auxiliary certificates, conjugators, classifications, parity witnesses, compatible bases, or local-to-global mechanisms. If an option bundles the desired conclusion with an explanatory mechanism, invariance claim, structural adjective, or compatibility condition, evaluate the added clause as part of the answer.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
