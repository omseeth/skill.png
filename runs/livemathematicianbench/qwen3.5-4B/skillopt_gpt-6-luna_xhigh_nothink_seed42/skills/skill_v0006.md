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
Before committing, translate each live option into its exact logical claim: its quantifiers, hypotheses, endpoints, and every extra conclusion. Treat each added clause in a stronger-sounding option as a separate obligation to prove. An example can establish an existential claim, but cannot establish a universal one; a universal claim must survive every admissible case. Do not continue revising a choice after a direct calculation or an exact theorem statement distinguishes the options unless you find a concrete error in that check.

Use remembered results only as leads, never as evidence. Do not invent citations, theorem formulations, exceptions, or definitions for named invariants. Match the theorem’s object, assumptions, conditioning, parameter range, and conclusion to the question before applying it. If you cannot establish that match, do not strengthen or weaken an option by analogy or by a claim that merely sounds familiar.

For uniform estimates with an unspecified constant, determine whether the constant may depend on dimension and other stated data before comparing polynomial factors; a small-dimensional value alone does not disprove a bound with some positive uniform constant. Conversely, do not add a lower bound, sharper rate, endpoint case, or structural conclusion to a theorem that only supplies a one-sided or weaker estimate. Preserve explicit coefficients and cutoffs when comparing quantitative estimates rather than dismissing them as nonstandard.

Keep these structural distinctions active: a full-rank lattice order inside a rational algebra need not be ring-isomorphic to that algebra, so do not infer that one cohomology ring realizes every order. Free subgroups of a finite-rank free group can have larger rank than the ambient group; use subgroup rank, not ambient rank, when assessing embeddings and equalizers. A rational projective image does not make its source point rational: require point separation or the exact stated theorem, and do not substitute finiteness or pure inseparability for that condition.

In parameter casework, derive feasible values from the original identity and every domain restriction before selecting a branch. In particular, retain constraints such as c-prime belonging to ell times the ell-adic integers, and distinguish the valuation of a quantity from the valuation of one plus that quantity. Do not classify a boundary case as possible just because the displayed definition appears to allow it.

For named, specialized results, resist generic intuition: use the stated theorem’s precise regime for fixed-size block trees and its threshold endpoint; do not infer a supercritical phase from generic branching-process terminology. Do not infer a knot-density cutoff from energy analogies, or a Monge-only restriction for a nondegenerate-manifold theorem without evidence. For Seifert matrices, derive the twist-induced matrix change under the specified convention and use the correct invariant; never guess a factor of two or confuse det(M) with det(M minus M-transpose). For multiple-zeta identities, copy every index block exactly, compute weights from the given definition, and test a small admissible specialization before relying on a remembered identity.
<!-- SLOW_UPDATE_END -->
