# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:45:27 2025

@author: cyril
"""




from matplotlib.patches import Ellipse
from scipy.stats import chi2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


c1 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvCP1sim.csv", sep=',')
c2 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvCP2sim.csv", sep=',')
c3 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvCP1.csv", sep=',')
c4 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvCP2.csv", sep=',')
c5 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvCP5.csv", sep=',')
c6 = pd.read_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvCP6.csv", sep=',')



jours_communs = set(c1['time']) & set(c2['time']) 

c11 = c1[c1['time'].isin(jours_communs)].copy()
c22 = c2[c2['time'].isin(jours_communs)].copy()



df=c11\
   .merge(c22, on='time')

df_normal = df[df['time'] < '2024-01-01']


# --- Données normales pour calibrer l'ellipsoïde ---
df_normal = df[df['time'] < '2024-01-01']

# Empilement en array 2D (n_points x 2) à partir des données normales
points_normal = np.vstack((df_normal['res_x'], df_normal['res_y'])).T

# Moyenne et covariance sur les données normales
mean = np.mean(points_normal, axis=0)
cov = np.cov(points_normal, rowvar=False)
inv_cov = np.linalg.inv(cov)

# --- Application de l'ellipsoïde sur toutes les données ---
points_all = np.vstack((df['res_x'], df['res_y'])).T
diff = points_all - mean
mahal_squared = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)

# p-values sur l'ensemble du jeu de données
df['pval'] = 1 - chi2.cdf(mahal_squared, df=2)

# Résultat sous forme de DataFrame
outside_df = df[df['pval'] < 0.05]
out1 = df[df['p_value_x'] < 0.05]
out2 = df[df['p_value_y'] < 0.05]
outtous = df[
    (df['p_value_y'] < 0.05) & 
    (df['p_value_x'] < 0.05) & 
    (df['pval'] < 0.05)
]

# Valeur de chi² pour 95 % de confiance avec 2 dimensions
chi2_val = chi2.ppf(0.95, df=2)

# Décomposition pour tracer l'ellipse
eigvals, eigvecs = np.linalg.eigh(cov)
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
width, height = 2 * np.sqrt(eigvals * chi2_val)
angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

# Intervalles de confiance univariés
from statistics import stdev
q025_c22 = df_normal['res_y'].mean() - 1.99 * stdev(df_normal['res_y'])
q975_c22 = df_normal['res_y'].mean() + 1.99 * stdev(df_normal['res_y'])
q025_c11 = df_normal['res_x'].mean() - 1.99 * stdev(df_normal['res_x'])
q975_c11 = df_normal['res_x'].mean() + 1.99 * stdev(df_normal['res_x'])




# Tracé
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(c11['res'], c22['res'], s=5, alpha=0.5)
ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor='red', facecolor='none', linewidth=2, label='Ellipse de confiance multivariée (95%)')
ax.add_patch(ellipse)


ax.scatter(out2['res_x'],out2['res_y'],color='gold',s=15)
ax.scatter(out1['res_x'],out1['res_y'],s=15,color='gold',label='Outliers univariés')
ax.scatter(outside_df['res_x'],outside_df['res_y'],s=15,color='orange',label='Outliers multivariés')
ax.scatter(outtous['res_x'], outtous['res_y'], s=25, color='red',label='Outliers univariés et multivariés')

plt.axvline(q025_c11, color='red', linestyle='--')
plt.axvline(q975_c11, color='red', linestyle='--')
plt.axhline(q025_c22, color='red', linestyle='--')
plt.axhline(q975_c22, color='red', linestyle='--', label='Intervalles de confiance univariés (95 %)')

#ax.set_xlim(-1.3, 1.3)
#ax.set_ylim(-1.8 ,1.8)

# Étiquettes
ax.set_xlabel("Résidus capteur CP2-01")
ax.set_ylabel("Résidus capteur CP2-02")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# --- Histogramme res_x ---
plt.subplot(1, 2, 1)
counts, bins, patches = plt.hist(df['res_x'], bins=50, alpha=0.7)

outlier_label_added = False
for patch, left, right in zip(patches, bins[:-1], bins[1:]):
    mid = (left + right) / 2
    if mid < q025_c11 or mid > q975_c11:
        patch.set_facecolor('orange')
        if not outlier_label_added:
            patch.set_label('Outliers')
            outlier_label_added = True
    else:
        patch.set_facecolor('steelblue')

plt.axvline(q025_c11, color='red', linestyle='--', label='2.5%')
plt.axvline(q975_c11, color='red', linestyle='--', label='97.5%')
plt.xlabel('Résidus capteurs CP2-01')

# Réorganiser la légende pour mettre Outliers en dernier
handles, labels = plt.gca().get_legend_handles_labels()
if "Outliers" in labels:
    idx = labels.index("Outliers")
    handles.append(handles.pop(idx))
    labels.append(labels.pop(idx))
plt.legend(handles, labels)

# --- Histogramme res_y ---
plt.subplot(1, 2, 2)
counts, bins, patches = plt.hist(df['res_y'], bins=50, alpha=0.7)

outlier_label_added = False
for patch, left, right in zip(patches, bins[:-1], bins[1:]):
    mid = (left + right) / 2
    if mid < q025_c22 or mid > q975_c22:
        patch.set_facecolor('orange')
        if not outlier_label_added:
            patch.set_label('Outliers')
            outlier_label_added = True
    else:
        patch.set_facecolor('steelblue')

plt.axvline(q025_c22, color='red', linestyle='--', label='2.5%')
plt.axvline(q975_c22, color='red', linestyle='--', label='97.5%')
plt.xlabel('Résidus capteurs CP2-03')

# Réorganiser la légende pour mettre Outliers en dernier
handles, labels = plt.gca().get_legend_handles_labels()
if "Outliers" in labels:
    idx = labels.index("Outliers")
    handles.append(handles.pop(idx))
    labels.append(labels.pop(idx))
plt.legend(handles, labels)

plt.tight_layout()
plt.show()




from statistics import stdev
mean_x = df['res_y'].mean()
std_x = stdev(df['res_y'])

test = df[(df['res_y'] < mean_x - 1.96 * std_x)] 
test2=df[(df['res_y'] > mean_x + 1.96 * std_x)] 







