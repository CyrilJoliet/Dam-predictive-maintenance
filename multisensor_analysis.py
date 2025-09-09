import pandas as pd
import numpy as np
from scipy.stats import chi2

# ─── Chargement des fichiers ────────────────────────────────────────────────────
c1 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvJ1.csv", sep=',')
c2 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvJ2.csv", sep=',')
c3 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvJ3.csv", sep=',')
c4 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvJ4.csv", sep=',')
c5 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvJ5.csv", sep=',')
c6 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvCP6sim.csv", sep=',')

# ─── Renommage pour éviter les collisions ──────────────────────────────────────
def renommer(df, suf):
    return df.rename(columns={col: f"{col}_{suf}" for col in df.columns if col != 'time'})

c1, c2, c3 = renommer(c1,'c1'), renommer(c2,'c2'), renommer(c3,'c3')
c4, c5 = renommer(c4,'c4'), renommer(c5,'c5')#, renommer(c6,'c6')

# ─── Fusion sur les dates communes ─────────────────────────────────────────────
jours = set(c1['time']) & set(c2['time']) & set(c3['time']) & set(c4['time']) & set(c5['time']) #& set(c6['time'])
dfs = [df[df['time'].isin(jours)] for df in (c1,c2,c3,c4,c5)]
df = dfs[0]
for d in dfs[1:]:
    df = df.merge(d, on='time')

# ─── Séparation données normales vs. complètes ─────────────────────────────────
df_normal = df[df['time'] < '2024-01-01']      # ↔ période saine
df_all    = df                                 # ↔ tout le jeu

# ─── Matrice des résidus (6 dim.) ──────────────────────────────────────────────
cols_res = ['res_c1','res_c2','res_c3','res_c4','res_c5']
X_normal = df_normal[cols_res].values          # → référence pour l’ellipse
X_all    = df_all[cols_res].values             # → à évaluer

# ─── Moyenne & covariance sur la période saine ─────────────────────────────────
mean_ref = X_normal.mean(axis=0)
cov_ref  = np.cov(X_normal, rowvar=False)
inv_cov  = np.linalg.inv(cov_ref)

# ─── Distance de Mahalanobis appliquée à toutes les dates ──────────────────────
diff        = X_all - mean_ref
mahal_sq    = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
df_all['pval'] = 1 - chi2.cdf(mahal_sq, df=5)

# ─── Outliers multivariés + combinaisons univariées ────────────────────────────
alpha = 0.05
outside_df = df_all[df_all['pval'] < alpha]

mask_uni = (df_all[[f'p_value_c{i}' for i in range(1,6)]] < alpha)
outtous1 = df_all[(mask_uni.sum(axis=1) >= 1) & (df_all['pval'] < alpha)]
outtous2 = df_all[(mask_uni.sum(axis=1) == 2) & (df_all['pval'] < alpha)]
outtous3 = df_all[(mask_uni.sum(axis=1) == 3) & (df_all['pval'] < alpha)]
outtous4 = df_all[(mask_uni.sum(axis=1) == 4) & (df_all['pval'] < alpha)]
outtous5 = df_all[(mask_uni.sum(axis=1) == 5) & (df_all['pval'] < alpha)]

