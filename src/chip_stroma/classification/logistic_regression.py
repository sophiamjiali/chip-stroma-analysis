# ==============================================================================
# Script:           logistic_regression.py
# Purpose:          Patient-level CHIP versus non-CHIP
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/03/2026
# ==============================================================================

def chip_vs_nonchip_test(density_df: pd.DataFrame, chip_labels: pd.DataFrame) -> dict:
    """t-test replication of wet-lab finding; returns stat, p-value, effect size."""
    ...