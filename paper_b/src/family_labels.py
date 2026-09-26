"""
Stoichiometric-template family labeling for Paper B.

Assigns family labels (e.g. ABX half-Heusler, AB3 skutterudite) from
composition alone via template matching, since Starrydata2 carries no
structure data. Unmatched compositions are routed to an explicit
"unassignable" bucket rather than dropped, with its size reported.
Restricts the downstream family study to families above the a priori
sample-size threshold and reports what fraction of the dataset this
template-matched subset covers.
"""
