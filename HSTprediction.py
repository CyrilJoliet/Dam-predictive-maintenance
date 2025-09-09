
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
import statsmodels.api as sm

import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns
warnings.filterwarnings("ignore")



def load_and_preprocess(file_path, start_date, end_date):
    df = pd.read_csv(file_path, sep=',')
    df=df.dropna()
    df["Time"] = pd.to_datetime(df["Timestamp"], format='mixed')
    df = df[(df["Time"] >= pd.Timestamp(start_date)) & (df["Time"] <= pd.Timestamp(end_date))]
    df["Time"] = df["Time"].dt.date
    df = df.drop_duplicates(subset="Time", keep="first")
    df["Day"] = (pd.to_datetime(df["Time"]) - pd.to_datetime(start_date)).dt.days
    df = df.sort_values(by="Day").reset_index(drop=True)  
    return df






data=load_and_preprocess(b1, "2010-01-01", "2026-01-01")
ref=load_and_preprocess(limni, "2010-01-01", "2026-01-01")
Te=load_and_preprocess(temp, "2010-01-01", "2026-01-01")
Th=load_and_preprocess(therm, "2010-01-01", "2026-01-01")

#data=data[data['Value']>620]


plt.plot(therm['Time'],therm['Value'], label="Hauteur d'eau dans la retenue")
plt.ylabel("Hauteru d'eau [m]")
plt.xlabel("Date")
plt.legend()
plt.show()


jour_range = pd.date_range(start=pd.to_datetime(Te['Time'].min()), 
                           end=pd.to_datetime(Te['Time'].max()) + pd.Timedelta(days=1), 
                           freq='D')
Te = Te.set_index('Time').reindex(jour_range, fill_value=0).reset_index()


Te['pcum']=(Te['Value'].rolling(window=90, min_periods=1).sum())
Te["index"]=Te["index"]+pd.Timedelta(days=1)
#data=data[data['Value']>100]

Te3=Te.copy()
data3=load_and_preprocess(b1, "2010-01-01", "2026-01-01")

#data3=data3[data3['Value']>620]


jours_communs = set(ref['Time']) & set(data['Time']) 

# Filtrer tous les DataFrames en une seule étape
ref = ref[ref['Time'].isin(jours_communs)].copy()
data = data[data['Time'].isin(jours_communs)].copy()

fig, ax1 = plt.subplots(figsize=(10, 5))

# --- Premier axe Y (piezométrie) ---
ax1.plot(Th['Time'], Th['Value'], color='steelblue', label="T°")
ax1.plot(ref['Time'], ref['Value']-650, color='orange', label='Précipitations')
ax1.set_xlabel('Temps')
ax1.set_ylabel("Hauteur d'eau [m]")
ax1.tick_params(axis='y')

# --- Second axe Y (pcum) ---
ax2 = ax1.twinx()

ax2.plot(Te['index'], Te['pcum'], color='orange', label='Précipitations')
ax2.set_ylabel('Précipitation [mm]')
ax2.tick_params(axis='y')

# --- Légendes combinées ---
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

plt.plot(Te['index'], Te['pcum'], color='steelblue', label='Précipitations')
plt.ylabel('Précipitation [mm]')
plt.xlabel('Temps')
data666=data3.copy()
#######SIMMMMMMMMMM

'''
# 1. Conversion de la colonne Time (si nécessaire)
data3['Time'] = pd.to_datetime(data3['Time'])

# 2. Définir la date de départ de la dérive
start_drift_date = pd.to_datetime("2024-01-01")

# 3. Créer un masque pour les dates à partir du 01/01/2022
mask = data3['Time'] >= start_drift_date

# 4. Créer une nouvelle colonne temporaire pour les jours écoulés depuis le 1er janv 2022
data3['days_since'] = 0  # initialisation

# 5. Calculer les jours écoulés uniquement là où le masque est vrai
data3.loc[mask, 'days_since'] = (data3.loc[mask, 'Time'] - start_drift_date).dt.days

# 6. Appliquer la dérive à 'Value'
data3.loc[mask, 'Value'] += (2/ 365) * data3.loc[mask, 'days_since']

data3["Time"] = data3["Time"].dt.date

plt.plot(data3['Time'], data3['Value'])
plt.show()

'''
df1 = pd.DataFrame({
    'Time': ref['Time'].reset_index(drop=True),
    'data': data['Value'].reset_index(drop=True),
    'Day': ref['Day'].reset_index(drop=True),
    'ref': ref['Value'].reset_index(drop=True)
})

# Définir les valeurs de Hmax, Hmin, Hi, t
Hmax = df1["ref"].max()
Hmin = df1["ref"].min()
Hi = df1["ref"]
t = df1["Day"]
y_target = df1["data"]


# Calcul de Z
Z = (Hmax - Hi) / (Hmax - Hmin)
w = (2 * np.pi) / 365.25



#%% HST base

X = np.column_stack([
    np.ones(len(t)),  
    Z,               
    Z**2,             
    Z**3,             
    np.sin(w * t),    
    np.cos(w * t),    
    np.sin(2*w * t),
    np.cos(2*w*t),
    np.log((t+1)),
    t
   
])


model = sm.OLS(y_target, X)  # OLS : Moindres Carrés Ordinaires
HST = model.fit()

# Résultats de la régression
print(HST.summary()) 







a=0.05
confidence_level = (1 - a) * 100
predictions = HST.get_prediction(X)


conf_int = predictions.conf_int(alpha=a)
lower_bound = conf_int[:, 0]
upper_bound = conf_int[:, 1]


summary = predictions.summary_frame(alpha=a)  # alpha=0.05 → intervalle à 95%

# Extraire l'intervalle de prédiction
y_pred_hst = summary["mean"]
pred_lower = summary["obs_ci_lower"]  # Limite inférieure
pred_upper = summary["obs_ci_upper"]  # Limite supérieure


#df_X = pd.DataFrame(X, columns=["const", "Z", "Z^2","Z^3", "sin(wt)", "cos(wt)","sin2","cos2", "log(t+1)", "t"])

# Calcul des VIFs
#vif_data = pd.DataFrame()
#vif_data["Variable"] = df_X.columns
#vif_data["VIF"] = [variance_inflation_factor(df_X.values, i) for i in range(df_X.shape[1])]

#print(vif_data)



cov_matrix = HST.cov_params()  # Matrice de variance-covariance des paramètres
correlation_matrix = cov_matrix / np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))


correlation_matrix.columns = ['Intercept', 'Z', 'Z²','Z³','sin(wt)','cos(wt)','sin(2wt)','cos(2wt)','log','t']
correlation_matrix.index = ['Intercept', 'Z', 'Z²','Z³','sin(wt)','cos(wt)','sin(2wt)','cos(2wt)','log','t']

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=30)  
plt.yticks(rotation=30) 
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=True)

# 1. Effet temps brut + log(t)
axs[1, 0].scatter(df1['Time'], y_target, s=3, color='gold', label="Measures")
axs[1, 0].plot(df1['Time'], HST.params[9] * t + HST.params[0]+HST.params[8]*np.log(t) , color='blue', label="T")
axs[1, 0].set_ylabel("Piezometry [m]")
axs[1, 0].set_title("Time component",fontsize=14)
axs[1, 0].legend(fontsize=14,markerscale=4)
axs[1, 0].show()
# 2. Effet Z polynomial
axs[0, 0].scatter(df1['Time'], y_target, s=3, color='gold', label="Measures")
axs[0, 0].plot(df1['Time'], HST.params[0] + HST.params[1] * Z + HST.params[2] * Z**2 + HST.params[3] * Z**3, color='green', label="H")
axs[0, 0].set_title("Hydrostatic component",fontsize=14)
axs[0, 0].legend(fontsize=14,markerscale=4)

# 3. Effets périodiques
axs[0, 1].scatter(df1['Time'], y_target, s=3, color='gold', label="Measures")
axs[0, 1].plot(df1['Time'], HST.params[0]
              + HST.params[4] * np.sin(w * t)
              + HST.params[5] * np.cos(w * t)
              + HST.params[6] * np.sin(2 * w * t)
              + HST.params[7] * np.cos(2 * w * t), color='purple', label="S")
axs[0, 0].set_ylabel("Piezometry [m]")
axs[0, 1].set_title("Seasonnal component",fontsize=14)
axs[0, 1].legend(fontsize=14,markerscale=4)

# 4. Prédiction complète
axs[1, 1].scatter(df1['Time'], y_target, s=3, color='gold', label="Measures")
axs[1, 1].plot(df1['Time'], HST.fittedvalues, color='red', label="HST")
axs[1, 1].set_title("Complete HST model",fontsize=14)
axs[1, 1].legend(fontsize=14,markerscale=4)

# Axes
for ax in axs.flat:
    ax.set_xlabel("Date")

plt.tight_layout()
plt.show()

#%% HST+pluie
Te['Time']=Te['index'].dt.date
jours = set(ref['Time']) & set(data['Time'])  & set(Th['Time']) & set(Te['Time'])


# Filtrer tous les DataFrames en une seule étape
ref3 = ref[ref['Time'].isin(jours)].copy()
data2 = data[data['Time'].isin(jours)].copy()
Te1 = Te[Te['index'].isin(jours)].copy()
Th1 = Th[Th['Time'].isin(jours)].copy()

df2 = pd.DataFrame({
    'Time': ref3['Time'].reset_index(drop=True),
    'data': data2['Value'].reset_index(drop=True),
    'Day': ref3['Day'].reset_index(drop=True),
    'ref': ref3['Value'].reset_index(drop=True),
    'plu':Te1['pcum'].reset_index(drop=True),
    'T':Th1['Value'].reset_index(drop=True)
})

np.corrcoef(df2['ref'],df2['plu'])

Hi = df2["ref"]
t2 = df2["Day"]
# Calcul de Z
Z2 = (Hmax - Hi) / (Hmax - Hmin)


# Construction de la matrice des features X
X1 = np.column_stack([ 
    np.ones(len(t2)),
    Z2,               
    Z2**2,             
    Z2**3,             
    np.sin(w * t2),    
    np.cos(w * t2),    
    np.sin(2*w * t2),
    np.cos(2*w*t2),
    np.log((t2+1)),
    #np.exp((t2+1)/t2.max()),
    t2,     
    df2['plu'],
    df2['T']
])



model = sm.OLS(df2['data'], X1)  # OLS : Moindres Carrés Ordinaires
pluvio = model.fit()


# Résultats de la régression
print(pluvio.summary()) 

ref2=load_and_preprocess(limni, "2017-01-01", "2026-01-01")

Th3=load_and_preprocess(therm, "2017-01-01", "2026-01-01")

Te3['Time']=Te3['index'].dt.date

jour = set(ref2['Time']) & set(Th3['Time']) & set(Te3['Time']) & set(data3['Time'])

ref4 = ref2[ref2['Time'].isin(jour)].copy()
Te2 = Te[Te['index'].isin(jour)].copy()
Th2 = Th3[Th3['Time'].isin(jour)].copy()
data3=data3[data3['Time'].isin(jour)].copy()
data666=data666[data666['Time'].isin(jour)].copy()


df3 = pd.DataFrame({
    'Time': ref4['Time'].reset_index(drop=True),
    'Day': ref4['Day'].reset_index(drop=True),
    'ref': ref4['Value'].reset_index(drop=True),
    'plu':Te2['pcum'].reset_index(drop=True),
    'T':Th2['Value'].reset_index(drop=True),
    'data':data3['Value'].reset_index(drop=True)
})



Hi = df3["ref"]
t2 = df3["Day"]
# Calcul de Z
Z2 = (Hmax - Hi) / (Hmax - Hmin)

# Construction de la matrice des features X
X3 = np.column_stack([ 
    np.ones(len(t2)),
    Z2,               
    Z2**2,             
    Z2**3,             
    np.sin(w * t2),    
    np.cos(w * t2),    
    np.sin(2*w * t2),
    np.cos(2*w*t2),
    np.log((t2+1)),
    t2,     
    df3["plu"],
    df3['T']
])


predictions = pluvio.get_prediction(X3)


conf_int = predictions.conf_int(alpha=a)
lower_bound = conf_int[:, 0]
upper_bound = conf_int[:, 1]


summary = predictions.summary_frame(alpha=a)  # alpha=0.05 → intervalle à 95%

# Extraire l'intervalle de prédiction
y_pred_hst2 = summary["mean"]
pred_lower2 = summary["obs_ci_lower"]  # Limite inférieure
pred_upper2= summary["obs_ci_upper"]  # Limite supérieure
#is_outside = (df2["data"].values < summary['obs_ci_lower'].values) | \
 #            (df2["data"].values > summary['obs_ci_upper'].values)
#is_outside.sum()

zizi = pd.DataFrame(X3, columns=['Intercept', 'Z', 'Z²','Z³','sin(wt)','cos(wt)','sin(2wt)','cos(2wt)','log','t','p','th'])  # donne un nom aux colonnes
correlation_matrix = zizi.drop(columns='Intercept').corr()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Tes groupes (les indices des variables dans l'ordre)
groups = {
    "Hydrostatique": ['Z', 'Z²', 'Z³'],
    "Saisonnière": ['sin(wt)', 'cos(wt)', 'sin(2wt)', 'cos(2wt)'],
    "Temporelle": ['log', 't'],
    "Pluviométrique": ['p'],
    "Thermique": ['th'],
}

# Ordre des variables
order = groups["Hydrostatique"] + groups["Saisonnière"] + groups["Temporelle"] + groups["Pluviométrique"] + groups["Thermique"]
corr_sorted = correlation_matrix.loc[order, order]

# Couleurs
colors = ['#FFD700', '#90EE90', '#87CEFA', 'red', 'green']

plt.figure(figsize=(12, 9))
ax = sns.heatmap(corr_sorted, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, alpha=0.9)

# === Groupes sur l'axe Y (gauche) ===
start_idx = 0
for i, (group_name, variables) in enumerate(groups.items()):
    length = len(variables)

    # Fond coloré
    ax.add_patch(patches.Rectangle(
        (-1, start_idx),
        width=1, height=length,
        facecolor=colors[i], edgecolor='none', alpha=0.3,
        transform=ax.transData, clip_on=False
    ))
    # Bordure noire
    ax.add_patch(patches.Rectangle(
        (-3, start_idx),
        width=3, height=length,
        facecolor='none', edgecolor='black', linewidth=1.5,
        transform=ax.transData, clip_on=False
    ))
    # Texte vertical (gauche)
    ax.text(-2.9, start_idx + length / 2, group_name,
            va='center', ha='left', fontsize=12)

    start_idx += length

# === Groupes sur l'axe X (bas) ===
start_idx = 0
for i, (group_name, variables) in enumerate(groups.items()):
    length = len(variables)

    # Fond coloré sous la heatmap
    ax.add_patch(patches.Rectangle(
        (start_idx, len(order)),
        width=length, height=1.5,
        facecolor=colors[i], edgecolor='none', alpha=0.3,
        transform=ax.transData, clip_on=False
    ))
    # Bordure noire
    ax.add_patch(patches.Rectangle(
        (start_idx, len(order)),
        width=length, height=4.1,
        facecolor='none', edgecolor='black', linewidth=1.5,
        transform=ax.transData, clip_on=False
    ))
    # Texte horizontal (bas)
    ax.text(start_idx + length / 2, len(order) + 3.9,
            group_name, ha='center', va='bottom', fontsize=12, rotation=90)

    start_idx += length

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





from scipy.stats import norm
import numpy as np

# 1. Obtenir les prédictions
summarty = pluvio.get_prediction(X3).summary_frame(alpha=0.05)

# 2. Valeurs observées (correspondant aux lignes de X3)
y_obs = df3["data"] 
# 3. Calcul de l'erreur standard de prédiction à partir de l'intervalle
#    obs_ci_upper = mean + 1.96 * SE_pred  =>  SE_pred = (upper - mean) / 1.96
SE_pred = (summarty['obs_ci_upper'] - summarty['mean']) / norm.ppf(1 - 0.05/2)  # pour alpha=0.05

# 4. Calcul du z-score
z_scores = (y_obs - summarty['mean']) / SE_pred

# 5. p-valeur bilatérale
p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

# 6. Tu peux maintenant afficher ou filtrer
summarty['y_obs'] = y_obs
summarty['z_score'] = z_scores
summarty['p_value'] = p_values
summarty['time'] = df3['Time']
summarty['res']= summarty['y_obs']-summarty['mean']
summarty['reel']=data666['Value'].values
# Exemple : observations avec p < 0.05
anomalies = summarty[summarty['p_value'] < 0.05]




plt.figure()
plt.scatter(summarty['time'], summarty['p_value'], s=3, label='p-value')
plt.axhline(0.05, color='red', linestyle='--', label='Seuil 0.05')
plt.xlabel("Time")
plt.ylabel("p-value")
plt.legend(fontsize=12)

plt.show()


plt.scatter(summarty["time"], summarty['y_obs'], label="Mesure",color="gold",s=2)
plt.plot(summarty["time"],summarty['mean'] ,label=f"Prédictions HST (R² = {pluvio.rsquared:.3f})", color="green",linestyle="dashed")
plt.fill_between(summarty['time'], summarty['obs_ci_lower'], summarty['obs_ci_upper'], color="green", alpha=0.2, label=f"Intervalle de prédiction ({confidence_level:.0f}%)")
plt.scatter(summarty['time'],summarty['reel'])
plt.scatter(data3['Time'],data3['Value'],s=10)


plt.scatter(anomalies["time"],anomalies["y_obs"],s=5,color='red',label='Anomalies')


plt.legend(fontsize=9)
plt.show()
summarty.to_csv(r"C:\Users\cyril\OneDrive\Documents\cours\memoire\data\pvJ6.csv", index=False)

plt.hist(summarty['p_value'], bins=50)


plt.hist(summarty['res'],bins=50)

sm.qqplot(summarty['res'], line='45', fit=True)

plt.show()



from scipy.stats import shapiro

stat, p_value = shapiro(pluvio.resid)
print(f"Statistique de Shapiro-Wilk : {stat:.4f}")
print(f"p-value : {p_value:.7f}")

if p_value > 0.05:
    print("✅ Les résidus suivent une loi normale (p > 0.05)")
else:
    print("❌ Les résidus ne suivent pas une loi normale (p ≤ 0.05)")
    
    
    
    

#%%BRT lol


#%% PLOT

fig, ax = plt.subplots(figsize=(10, 5))  # Créer une figure et un axe


ax.scatter(data3["Time"], data3['Value'], label="Mesure",color="gold",s=2)

ax.plot(df1["Time"],y_pred_hst ,label=f"HST (R² = {HST.rsquared:.3f})", color="purple",linestyle="dashed")
#ax.fill_between(ref2["Time"], pred_lower, pred_upper, color="green", alpha=0.2, label=f"Intervalle de prédiction ({confidence_level:.0f}%)")

ax.plot(summarty['time'],summarty['mean'],label=f"Improved HST (R² = {pluvio.rsquared:.3f})", color="green",linestyle="dashed")
#ax.fill_between(df2['Time'], pred_lower2, pred_upper2, color="green", alpha=0.2, label=f"Intervalle de prédiction ({confidence_level:.0f}%)")
#ax.scatter(anomalies["time"],anomalies["y_obs"])
#plt.scatter(outliers_ocsvm['Time'],outliers_ocsvm['c11'],s=3,color='red')
#ax.plot(ref["Time"], HST.fittedvalues, color="green",label="Prédictions ",linestyle="dashed")
#ax.plot(ref3["Time"], brt_model.predict(X_new_scaled), label=f"Prédictions BRT (R²={r2:.4f}", color="red", linestyle="dashed")


# Formatter les dates sur l'axe X
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format YYYY-MM-DD
#ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Auto-ajustement des labels
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.ylabel("Niveau piézometrique [m]")
plt.xlabel("Date")

#ax.set_xlim(pd.to_datetime("2017-01-01"), pd.to_datetime("2018-01-01"))
#ax.set_ylim(640,665)

plt.legend(fontsize=12,markerscale=4)
#plt.xticks(rotation=45)  # Rotation pour lisibilité

plt.show()








