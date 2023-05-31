from datetime import timedelta

def true_positives(T, X, margin=timedelta(hours=12)):
    """Compute true positives without double counting

    >>> true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> true_positives(set(), {1, 2, 3})
    set()
    >>> true_positives({1, 2, 3}, set())
    set()

    Modified from: G.J.J. van den Burg, Copyright (c) 2020 - The Alan Turing Institute
    """
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if timedelta() <= tau - x <= margin] # Consider leading indicators up to `margin`
        close.sort()
        if not close:
            continue
        dist, xstar = close[-1] # Take the farthest one (i.e. the most-leading indicator)
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=timedelta(hours=12), alpha=0.5, return_PR=False):
    """Compute the F-measure based on human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too

    Remember that all CP locations are 0-based!

    >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> f_measure({1: [], 2: [10], 3: [50]}, [])
    0.8

    Modified from: G.J.J. van den Burg, Copyright (c) 2020 - The Alan Turing Institute
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}

    X = set(predictions)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    if P == 0 and R == 0:
        F = 0 # avoid division by 0
    else:
        F = P * R / (alpha * R + (1 - alpha) * P)

    if return_PR:
        return F, P, R
    return F
