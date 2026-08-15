"""
Composition canonicalization and chemistry-cluster definition.

Canonicalizes raw formulas to a comparable composition representation,
then derives the frozen chemistry-cluster grouping key: reduced
host-lattice stoichiometry with dopants below the 5 at% threshold
collapsed into the parent. Also supports the looser/stricter threshold
variants used for the 3-row sensitivity table.

Establishes the Chemistry cluster >= Composition >= Sample hierarchy
used by every grouped split in the project.
"""
