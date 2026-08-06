# ==============================================================================
# Script:           statistics.py
# Purpose:          Unpaired t-test replication and effect size
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve


def run_loocv_auc(X_raw, y, n_permutations = 2000):
    """
    LOOCV logistic regression AUC with fixes:
    - StandardScaler fit per-fold (prevents scale-driven instability on small n)
    - Explicit class-order assertion (defends against class ordering assumptions)
    - Permutation test for empirical significance (small-n AUC needs this, not just point estimate)
    """

    loo_probs = np.zeros(len(y))
    for train_idx, test_idx in LeaveOneOut().split(X_raw):

        # Fit StandardScaler per fold
        scaler  = StandardScaler().fit(X_raw[train_idx])
        X_train = scaler.transform(X_raw[train_idx])
        X_test  = scaler.transform(X_raw[test_idx])

        # Fit logistic regression to the scaled data 
        clf = LogisticRegression().fit(X_train, y[train_idx])
        assert list(clf.classes_) == [0, 1], f"unexpected class order: {clf.classes_}"

        loo_probs[test_idx] = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y, loo_probs)
    fpr, tpr, thresh = roc_curve(y, loo_probs)

    # Permutation test: shuffle labels, recompute AUC, build null distribution
    rng = np.random.default_rng(0)
    null_aucs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        try: null_aucs.append(roc_auc_score(y_perm, loo_probs))
        except ValueError: continue

    perm_p = (np.mean(np.array(null_aucs) >= auc) if auc >= 0.5 
              else np.mean(np.array(null_aucs) <= auc))

    # Youden's J optimal cutpoint — clinically interpretable operating point
    j_scores      = tpr - fpr
    best_idx      = np.argmax(j_scores)
    youden_cutoff = thresh[best_idx]
    sens, spec    = tpr[best_idx], 1 - fpr[best_idx]

    return dict(
        auc           = auc,
        fpr           = fpr,
        tpr           = tpr,
        loo_probs     = loo_probs,
        perm_p        = perm_p,
        youden_cutoff = youden_cutoff,
        sens          = sens,
        spec          = spec
    )


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / pooled_sd * (1 - (3 / (4*(n1+n2) - 9)))

def bootstrap_ci(a, b, n_boot=10000):
    diffs = [np.random.choice(a, len(a)).mean() - np.random.choice(b, len(b)).mean() for _ in range(n_boot)]
    return np.percentile(diffs, [2.5, 97.5])

# [END]