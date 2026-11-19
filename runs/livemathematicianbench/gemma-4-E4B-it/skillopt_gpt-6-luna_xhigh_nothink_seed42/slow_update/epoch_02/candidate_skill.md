# Mathematical Multiple-Choice Skill

(No learned rules yet. Rules will be added through the reflection process.)

<!-- SLOW_UPDATE_START -->
Treat “strongest” as the strongest statement actually proved under the stated hypotheses, not the strongest one that seems plausible. Parse each option literally: track its quantifiers, endpoint inclusion, constant dependence, and whether it asserts an upper bound or an asymptotic equivalence. Do not upgrade a theorem without proof: compatibility conditions alone do not justify replacing C((0,T)) by C([0,T]); a fixed-time C_t ε bound is not uniform over all t; and an O-bound does not establish a matching lower bound or a ∼ claim.

For specialized named results, do not invent definitions or select a theorem by analogy with a familiar topic. Identify the exact result or derive what you can from the given definitions and hypotheses. When notation or context is incomplete, do not confidently rule out an option using a guessed interpretation; use stated parameter ranges to check and simplify candidate conditions. Keep distinct properties distinct: countability may characterize 1⁺-projectivity without characterizing semiprojectivity, and point separation is not interchangeable with finiteness onto an image. Verify the actual sufficient condition in the theorem rather than choosing by an informal ranking of hypotheses.

Check formulas against the wording and conventions before committing. If a(p) counts anchors away from the origin, do not add the origin again: the anchored-weight denominator is a(p). Derive the ruin exponent from the model or its governing equation rather than guessing between γ and γ−1: here the asymptotic power is u^(1−γ). For congruence conditions, use the full modulus and the stated range; for H(n,m), the condition is n ≡ 0 (mod 6) and m ≡ 2 (mod n), which with 0 ≤ m < n forces m = 2.

When these result patterns recur, preserve their exact scope: the IBVP regularity is C((0,T)), not an assumed endpoint upgrade; the global homogenization estimate includes ε[C + (n/2) log(max{t,ε}/ε)], so do not replace it with a fixed-time estimate; the adelic conclusion requiring rationality uses the additional point-separation condition, not merely finiteness onto the image; and the hypercube variation supremum is unbounded for every finite r, so do not import an r > 2 threshold from other variation problems. For the random-graph cop number, state the high-probability O(√n log n) upper bound unless a matching lower bound is established.
<!-- SLOW_UPDATE_END -->
