# ==============================================================================
# Script:           statistics.py
# Purpose:          Unpaired t-test replication and effect size
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

import numpy as np

def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
    d = (a.mean() - b.mean()) / pooled_sd
    return d * (1 - (3 / (4*(n1+n2) - 9)))

def bootstrap_ci(a, b, n_boot=10000):
    diffs = [np.random.choice(a, len(a)).mean() - np.random.choice(b, len(b)).mean() for _ in range(n_boot)]
    return np.percentile(diffs, [2.5, 97.5])

# [END]