"""
Composition-based descriptor featurization.

Computes descriptors once per unique formula rather than per row
(~13x compute reduction), via matminer composition-based featurizers.
CPU only, no GPU required.
"""
