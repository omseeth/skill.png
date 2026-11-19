# Mathematical Multiple-Choice Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Match theorem claims to the exact setup
- Do not choose an asymptotic exponent, regularity level, branching regime, or other conclusion from analogy alone. Verify the exact result under the stated hypotheses and for the stated object; distinctions such as conditioned versus unconditioned laws can change the conclusion.
- Treat a sharper rate, quantitative estimate, minimality property, changed parameter placement, or threshold behavior as a separate claim requiring support. When choices differ by an exponent shift or equality versus strict inequality, check the exact clause rather than preferring a plausible stronger or weaker claim.
- For algorithm bounds, distinguish a fixed-parameter form f(k)n^c from n^{O(k)}: polynomial time for each fixed k alone does not establish FPT.

## Check structural assumptions and feasibility
- Before using a special case or counterexample to reject a claim, verify that it satisfies every structural assumption in the question, such as nondegeneracy, regularity, parameter restrictions, and the stated measure. A case that violates a hypothesis does not refute the claim.
- Before declaring a parameter case possible or impossible, derive its range from all defining identities and hypotheses; variables in a displayed definition need not vary independently. In valuation cases, use the given equation as well as the stated valuation domains.
- Do not assume a subgroup has rank at most that of its ambient group: subgroups of a finite-rank free group can have larger rank. Distinguish subgroup rank from free-factor rank.

## Keep pointwise bounds separate from norm bounds
A pointwise estimate in terms of \(\|f\|_\infty\) does not by itself give a dimension-free \(\ell^2\) operator bound; converting pointwise control to an \(\ell^2\) norm can introduce a factor depending on the size of the domain. Analyze the requested norm and its dependence on dimension directly.

## Audit logical and formula details
- Translate each option into its logical form before comparing it. Track quantifier order and dependencies: “for every target, there exists a witness” differs from one witness working for every target, and a constant that may depend on a parameter differs from a uniform constant. Distinguish all parameter values from sufficiently large values, and a whole class from a restricted subclass.
- Preserve the exact form of asymptotic assertions: note cutoffs, the order of iterated limits, and whether the claim uses a limit or limsup. Do not silently replace a truncated approximation with convergence to a full series.
- For an equivalence, check both implications.
- Read negated restrictions literally: an option that says not to impose a condition may include cases excluded by an option that imposes it.
- For competing formulas, compare signs, coefficients, factors, index ranges, and wording such as “only one” versus “at most one.” Use definitions to check which terms and indices should appear; when useful, test a low-dimensional case, specialization, or boundary case rather than relying on plausibility.
- Separate a claimed conclusion from an offered explanation: a correct conclusion does not by itself establish the proposed mechanism. Once a direct check distinguishes the options, commit rather than adding speculative exceptions.

<!-- SLOW_UPDATE_START -->
When a question depends on a specialized theorem, do not reconstruct it from general intuition, a remembered title, or a plausible-sounding exception. Match the exact object, hypotheses, parameter regime, measure, and conclusion to a result you can actually state reliably. Never invent citations or theorem formulations. If you cannot verify an extra clause, do not claim it.

Treat every quantitative strengthening as a separate claim: an asymptotic equivalence is not established by an upper bound, a smaller logarithmic factor is not automatically valid, and a precise coefficient or endpoint cannot be guessed from dimension or analogy. Likewise, do not infer that a structured limit loses a property merely because general limits can lose it; use the hypotheses of the particular compactness result. Choose the strongest conclusion that is proved, not the strongest one that sounds plausible.

Do not equate distinct logical forms without establishing the missing implication. In particular, do not replace limsup by liminf, or infer monotonicity or endpoint regularity unless it follows from the equation or theorem. An existence result for some regularity exponent does not give every exponent. Do not rank incomparable options by which sounds deeper: verify each claim and its scope before selecting the strongest supported one.

Do not identify an undefined specialized object from notation or surrounding-topic guesses. For global sections on a family, distinguish the total space from a fibre and compute the restriction map and its kernel before asserting a basis; do not promote an ambient monomial spanning set to an independent basis without checking. In arithmetic casework, derive feasible valuation cases from the full identity and all domain restrictions; equal valuations can cancel in a sum, and a formally imaginable case is not necessarily feasible under the given equation.

When a direct calculation or exact theorem distinguishes the candidates, commit to that result. Reopen it only when you can name a concrete error in the calculation or a hypothesis that fails; do not use extended speculation to manufacture a contrary exception.
<!-- SLOW_UPDATE_END -->
