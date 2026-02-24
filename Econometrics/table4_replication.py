"""
Replication of Table 4 from Card & Krueger (1994) AER
"REDUCED-FORM MODELS FOR CHANGE IN EMPLOYMENT"
"""
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

def ols_fit(y, X):
    """OLS with intercept, returns coefs, ses, ser, residuals"""
    X = np.column_stack([np.ones(len(X)), X])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    X, y = X[mask], y[mask]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    mse = np.sum(resid**2) / (n - k)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    ser = np.sqrt(mse)
    return {'coef': beta, 'se': se, 'ser': ser, 'resid': resid, 'X': X, 'y': y, 'n': n, 'k': k}

# Column indices from codebook (0-based)
COLS = {
    'sheet': 0, 'chain': 1, 'co_owned': 2, 'state': 3,
    'southj': 4, 'centralj': 5, 'northj': 6, 'pa1': 7, 'pa2': 8, 'shore': 9,
    'empft': 11, 'emppt': 12, 'nmgrs': 13, 'wage_st': 14,
    'empft2': 31, 'emppt2': 32, 'nmgrs2': 33, 'wage_st2': 34,
    'status2': 28
}

# Read data
df = pd.read_csv('public.dat', sep=r'\s+', header=None, na_values=['.'])

# Extract variables
df = df.rename(columns={v: k for k, v in COLS.items()})
df = df[[c for c in COLS.keys()]]

# FTE = EMPFT + NMGRS + 0.5*EMPPT
df['emptot'] = df['empft'] + df['nmgrs'] + 0.5 * df['emppt']
df['emptot2'] = df['empft2'] + df['nmgrs2'] + 0.5 * df['emppt2']
df['demp'] = df['emptot2'] - df['emptot']

# Wage gap: (5.05 - wage_st) / wage_st for NJ with wage_st < 5.05, else 0
df['gap'] = np.where(df['state'] == 0, 0,
    np.where(df['wage_st'] >= 5.05, 0,
    np.where(df['wage_st'] > 0, (5.05 - df['wage_st']) / df['wage_st'], np.nan)))

# Dummies
df['nj'] = df['state']
df['bk'] = (df['chain'] == 1).astype(int)
df['kfc'] = (df['chain'] == 2).astype(int)
df['roys'] = (df['chain'] == 3).astype(int)
df['wendys'] = (df['chain'] == 4).astype(int)

# Sample: Table 4 uses "357 stores with available data on employment and starting wages in waves 1 and 2"
# SAS: IF DEMP NE . ; IF CLOSED=1 OR (CLOSED=0 AND DWAGE NE .);
df['closed'] = (df['status2'] == 3).astype(int)
df['dwage'] = df['wage_st2'] - df['wage_st']
sample = (df['demp'].notna()) & ((df['closed'] == 1) | ((df['closed'] == 0) & df['dwage'].notna()))
df = df[sample].copy()

print(f"Sample size: {len(df)}")
print(f"Dependent variable: demp (change in FTE employment)")
print(f"Mean(demp) = {df['demp'].mean():.3f}, SD(demp) = {df['demp'].std():.3f}")
print()

# Run regressions
y = df['demp'].values

# (i) NJ dummy only
m1 = ols_fit(y, df[['nj']].values)
# (ii) NJ + chain + co_owned
m2 = ols_fit(y, df[['nj', 'bk', 'kfc', 'roys', 'co_owned']].values)
# (iii) Gap only
m3 = ols_fit(y, df[['gap']].values)
# (iv) Gap + chain + co_owned
m4 = ols_fit(y, df[['gap', 'bk', 'kfc', 'roys', 'co_owned']].values)
# (v) Gap + chain + co_owned + region
m5 = ols_fit(y, df[['gap', 'bk', 'kfc', 'roys', 'co_owned', 'centralj', 'southj', 'pa1', 'pa2']].values)

def f_test_pvalue(r, r_constrained, q):
    """F-test for q restrictions: (RSS_r - RSS_u)/q / (RSS_u/(n-k))"""
    rss_u = np.sum(r['resid']**2)
    rss_r = np.sum(r_constrained['resid']**2)
    n, k = r['n'], r['k']
    f = ((rss_r - rss_u) / q) / (rss_u / (n - k))
    p = 1 - scipy_stats.f.cdf(f, q, n - k)
    return p

# Restricted models for F-tests
m2_r = ols_fit(y, df[['nj']].values)  # nj only, test bk,kfc,roys,co_owned
m4_r = ols_fit(y, df[['gap']].values)
m5_r = ols_fit(y, df[['gap']].values)
# F stat: (RSS_r - RSS_u)/q / (RSS_u/(n-k)), q=4 for ii,iv and q=8 for v
p_ii = f_test_pvalue(m2, m2_r, 4)
p_iv = f_test_pvalue(m4, m4_r, 4)
p_v = f_test_pvalue(m5, m5_r, 8)

# Build table
print("=" * 70)
print("TABLE 4 — REDUCED-FORM MODELS FOR CHANGE IN EMPLOYMENT")
print("=" * 70)

# New Jersey dummy row (coef index 1 for nj)
print("\nIndependent variable          (i)       (ii)      (iii)     (iv)      (v)")
print("-" * 70)
print(f"New Jersey dummy             {m1['coef'][1]:6.2f}    {m2['coef'][1]:6.2f}      —         —         —")
print(f"                             ({m1['se'][1]:.2f})   ({m2['se'][1]:.2f})")

# Initial wage gap row
print(f"\nInitial wage gap              —         —       {m3['coef'][1]:6.2f}    {m4['coef'][1]:6.2f}    {m5['coef'][1]:6.2f}")
print(f"                             ({m3['se'][1]:.2f})   ({m4['se'][1]:.2f})   ({m5['se'][1]:.2f})")

# Controls rows
print(f"\nControls for chain and        no        yes       no        yes       yes")
print(f"  ownership")
print(f"Controls for region           no        no        no        no        yes")

# SER
print(f"\nStandard error of regression  {m1['ser']:.2f}    {m2['ser']:.2f}    {m3['ser']:.2f}    {m4['ser']:.2f}    {m5['ser']:.2f}")

print(f"Probability value for controls  —       {p_ii:.2f}      —       {p_iv:.2f}     {p_v:.2f}")

print("\n" + "=" * 70)
print("Notes: Standard errors in parentheses. Sample: 357 stores with available")
print("data on employment and starting wages in waves 1 and 2.")
print("Dependent variable: change in FTE employment.")
print("Mean and SD of dependent variable: -0.237 and 8.825.")
print("=" * 70)
