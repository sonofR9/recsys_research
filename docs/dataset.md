# yambda-50m

**10,000 users and 47.8M events** — the 50m counts interactions, not users, so
`UserSample(max_users=10_000)` selects the whole dataset rather than a subset.

| event | events | users | distinct items |
| --- | --- | --- | --- |
| listen | 46,467,212 | 9,238 | 877,168 |
| like | 881,456 | 8,283 | 181,304 |
| unlike | 312,972 | 6,406 | 117,953 |
| dislike | 107,776 | 5,951 | 53,413 |

Likes are 1.8% of the events. A likes-only experiment therefore trains on
~13.5k sliding-window sequences over a 181k-item catalog, and only **1,119
users are evaluable** on a single held-out day — a small enough denominator
that it, not the model, sets the resolution of every comparison.

The item id space is wider than any one experiment uses: compact ids run to
**847,921**, so an embedding table sized to the catalog carries rows for items
a likes-only run never touches.
