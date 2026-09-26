"""
Leave-one-family-out evaluation and controls for Paper B.

Runs true LOFO plus two controls: size-matched random-removal (scattered
rows, same total reduction) and structured-removal of a different whole
family of similar size. Also runs the primary specialist-vs-pooled probe
(one specialist model per qualifying family vs. the pooled model's
within-family performance) and the offset-stripped within-family
predicted-vs-actual correlation, across two model families of bracketing
capacity (one high-capacity GBDT, one constrained), reported separately.
Sample-level grouping is applied identically across true LOFO and both
controls.
"""
