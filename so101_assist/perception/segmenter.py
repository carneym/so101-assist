"""SAM 2 mask refinement.

Given a Detection box, produce a pixel mask. The localizer uses the
mask centroid (better than box center for irregular objects like keys)
and the planner uses the mask extent to pick a grasp axis.
Only run on the selected target, not every detection — it's expensive.
"""
