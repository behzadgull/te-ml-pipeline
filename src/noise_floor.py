"""
Noise-floor anchor, computed in log space.

Computes R^2_max = 1 - sigma^2_noise(log) / sigma^2_total(log), using
the Alleno et al. 2015 round-robin uncertainties as the noise reference
(S ~6%, sigma ~8%, kappa ~11%, zT ~17-19%) and this dataset's actual
log-property variance for sigma_total. Also reports model R^2 in log
space for direct comparability against the computed ceiling.
"""
