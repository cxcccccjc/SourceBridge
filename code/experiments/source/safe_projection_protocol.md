# Safe projection of a nominal estimate

Let H=[l,u] be the nonempty convex hull of the depth-feasible truth set, T the
public target interval, and W the purchased pre-query width. Set R=W/2 and

    S(H,R) = [u-R, l+R] intersect T.

For a nominal estimate z, the SourceBridge point is the projection of z onto S.
Under the model's stated contract, the target belongs to H, the hull width is
at most W, and every point in S is at distance at most R from every target in H.
The implementation evaluates the exact interval endpoints with rational
arithmetic. A machine-value export has a separately rounded conservative radius.
Empty hulls and undefined projections are reported explicitly.

safe_projection.py implements this interface independently of how z is obtained.
The nominal MLNI updates do not supply the deterministic certificate. Equality
between standalone proper MLNI and SourceBridge in the saved evaluation is an
observed property of those predictions, not an additional theorem.
