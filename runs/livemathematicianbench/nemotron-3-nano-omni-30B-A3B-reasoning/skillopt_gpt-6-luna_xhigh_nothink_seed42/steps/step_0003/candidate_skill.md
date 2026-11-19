# Mathematical Multiple-Choice Skill

## Exact comparison of answer choices
- Translate each option into a precise claim. Preserve its quantifiers, logical connectives, domain, conditions, parameter dependence, and qualifiers such as “for every,” “there exists,” and “if and only if.” Distinguish existence from a converse, one direction from equivalence, and strict bounds from inclusive ones.
- Apply the question’s hypotheses before comparing choices. Simplify conditions using the permitted ranges and boundary cases; differently worded conditions may be equivalent on the stated domain. If two choices reduce to the same proposition, recognize the ambiguity rather than inventing a distinction.
- Compare logical strength under the hypotheses and choose the strongest claim actually established. Do not settle for a weaker consequence when an exact or strongest result is requested, but do not add an unsupported converse, endpoint, equality, exception, or other strengthening. Retain any condition on a valid conditional conclusion rather than silently assuming it holds.
- Check distinctions such as universal claims versus claims about a special subclass or extendible objects; restricted-range versus full-range claims; local versus global results; eventual versus finite-time behavior; and interior-time versus endpoint regularity. Extra rates, coefficients, and other precise details require their own support.
- For quantitative bounds in the same direction, identify which threshold makes a claim stronger, then verify which threshold is actually proved. Do not select the largest displayed number merely because it is strongest-looking; compare constants and asymptotic bounds precisely.
- For generalized or nonstandard notions, do not import familiar classical conclusions without verifying that the relevant result and its hypotheses apply.

## Additional exactness checks
- Verify uniqueness separately from existence. Distinguish a claim that a property holds from a claim that explicit witnesses can be constructed.
- Preserve the stated mode of convergence or probability: positive probability does not imply almost-sure behavior, and similar-sounding modes of convergence are not interchangeable.
- For exact identities and formulas, check signs, coefficients, and term or index order; resemblance or asymptotic agreement does not establish equality.
- Do not infer a uniform eventual estimate with a fixed multiplicative constant at the same exponent from a liminf statement alone. Keep explicit sufficient conditions distinct from vague eventual claims.

## Parameterized runtime and regularity claims
- Normalize parameterized runtimes before classifying them: FPT has the form f(k)n^c with a constant exponent c independent of k, while XP permits n^g(k). For example, 2^{O(k log n)}n^{O(1)} = n^{O(k)} is generally XP, not FPT; 2^{O(k)}n^{O(1)} is FPT. Polynomial time for each fixed k alone does not establish a uniform FPT bound.
- For regularity claims, check which compatibility conditions the stated result requires. Do not invent extra conditions, and do not infer endpoint smoothness from interior smoothing alone.

## Constant dependence in estimates
- Check which data the constant in an estimate may depend on. A bound with a constant uniform in specified parameters is a distinct, potentially stronger claim than one with an unspecified constant.

- For an exact list or partition, verify both that every included case satisfies the claim and that every excluded case meets the stated failure condition. If the classification also claims attainability of outcomes, verify that separately; almost-sure exhaustiveness alone does not establish, for example, positive probability for each alternative.

## Scope and measure checks
- Keep generic, universal, and almost-everywhere claims distinct. For an almost-everywhere claim, preserve the underlying measure and its domain; a measure supported on a restricted locus need not be ambient Lebesgue measure.
- For existence claims into a specified ambient space, preserve its dimension and distinguish existence in the full space from existence subject to an added geometric constraint or prescribed construction.

<!-- SLOW_UPDATE_START -->
<!-- SLOW_UPDATE_END -->
