"""
Pas 4: Decision Boundaries
Clasificarea tipurilor de turiști și vizualizarea decision boundaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("ANALIZA DECISION BOUNDARIES - FLUXURI TURISTICE MOLDOVA")
print("=" * 80)

df = pd.read_csv('output/dataset_clean.csv', index_col=0)
print(f"\n📊 Dataset încărcat: {df.shape}")

# Clasificare țări în categorii
cis_countries = ['Armenia', 'Azerbaijan', 'Belarus', 'Georgia (CIS)', 'Kazakhstan', 
                 'Kyrgyzstan', 'Russian Federation', 'Tajikistan', 'Turkmenistan', 
                 'Ukraine', 'Uzbekistan']

european_countries = ['Albania', 'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus',
                      'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
                      'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
                      'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Spain',
                      'Sweden', 'Switzerland', 'United Kingdom', 'Czech Republic', 'Slovenia',
                      'Slovakia', 'Malta', 'Iceland', 'Liechtenstein', 'Montenegro',
                      'North Macedonia', 'Serbia', 'Bosnia and Herzegovina']

# Caracteristici pentru clasificare
features_df = pd.DataFrame(index=df.index)
features_df['total_tourists'] = df.sum(axis=1)
features_df['avg_yearly'] = df.mean(axis=1)
features_df['std_yearly'] = df.std(axis=1)
features_df['recent_avg'] = df[['2020', '2021', '2022', '2023', '2024']].mean(axis=1)
features_df['early_avg'] = df[['1992', '1993', '1994', '1995', '1996']].mean(axis=1)
features_df['peak_year'] = df.idxmax(axis=1).astype(int)
features_df['participation_rate'] = (df > 0).sum(axis=1) / len(df.columns)

# Clasificare
def classify_country(country):
    if country in cis_countries:
        return 0  # CIS
    elif country in european_countries:
        return 1  # Europa
    else:
        return 2  # Altele

features_df['category'] = features_df.index.map(classify_country)
features_df = features_df.dropna()

# Filtrare țări cu activitate minimă
features_df_active = features_df[features_df['total_tourists'] > 100].copy()

print(f"\n📊 Țări active (>100 turiști): {len(features_df_active)}")
print(f"   CIS: {(features_df_active['category'] == 0).sum()}")
print(f"   Europa: {(features_df_active['category'] == 1).sum()}")
print(f"   Altele: {(features_df_active['category'] == 2).sum()}")

# ====================================================================================
# PLOT 1: PCA și Decision Boundaries
# ====================================================================================
X = features_df_active[['total_tourists', 'avg_yearly', 'std_yearly', 'recent_avg', 
                         'early_avg', 'participation_rate']].values
y = features_df_active['category'].values

# Normalizare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA pentru vizualizare 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Scatter plot cu categorii
ax1 = axes[0, 0]
colors = ['#E74C3C', '#3498DB', '#2ECC71']
labels = ['CIS', 'Europa', 'Altele']

for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, 
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11, fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11, fontweight='bold')
ax1.set_title('Spațiu de Caracteristici (PCA)', fontsize=12, fontweight='bold', pad=20)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Decision Tree boundaries
ax2 = axes[0, 1]
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_pca, y)

# Creare mesh grid
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.3, colors=colors, levels=[-0.5, 0.5, 1.5, 2.5])
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, 
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

ax2.set_xlabel(f'PC1', fontsize=11, fontweight='bold')
ax2.set_ylabel(f'PC2', fontsize=11, fontweight='bold')
ax2.set_title('Decision Boundaries: Decision Tree', fontsize=12, fontweight='bold', pad=20)
ax2.legend(fontsize=10)

# SVM boundaries
ax3 = axes[1, 0]
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_pca, y)

Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

ax3.contourf(xx, yy, Z_svm, alpha=0.3, colors=colors, levels=[-0.5, 0.5, 1.5, 2.5])
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, 
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

ax3.set_xlabel(f'PC1', fontsize=11, fontweight='bold')
ax3.set_ylabel(f'PC2', fontsize=11, fontweight='bold')
ax3.set_title('Decision Boundaries: SVM (RBF kernel)', fontsize=12, fontweight='bold', pad=20)
ax3.legend(fontsize=10)

# K-Means clustering
ax4 = axes[1, 1]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
            s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax4.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroizi')
ax4.set_xlabel(f'PC1', fontsize=11, fontweight='bold')
ax4.set_ylabel(f'PC2', fontsize=11, fontweight='bold')
ax4.set_title('Clustering K-Means (k=3)', fontsize=12, fontweight='bold', pad=20)
ax4.legend(fontsize=10)

plt.tight_layout()
plt.savefig('output/12_decision_boundaries.png', dpi=300, bbox_inches='tight')
print("\n✅ Salvat: output/12_decision_boundaries.png")
plt.close()

# ====================================================================================
# PLOT 2: Heatmap caracteristici pe categorii
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Heatmap medii caracteristici
ax1 = axes[0, 0]
features_by_category = features_df_active.groupby('category')[['total_tourists', 'avg_yearly', 
                                                                 'std_yearly', 'recent_avg', 
                                                                 'early_avg', 'participation_rate']].mean()
features_by_category.index = ['CIS', 'Europa', 'Altele']
sns.heatmap(features_by_category.T, annot=True, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': 'Valoare'}, ax=ax1, linewidths=0.5)
ax1.set_title('Caracteristici Medii pe Categorii', fontsize=12, fontweight='bold', pad=20)
ax1.set_ylabel('Caracteristică', fontsize=10, fontweight='bold')

# Box plots per caracteristică
ax2 = axes[0, 1]
data_for_box = []
for cat in [0, 1, 2]:
    data_for_box.append(features_df_active[features_df_active['category']==cat]['total_tourists'].values)
bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_ylabel('Total turiști (log scale)', fontsize=10, fontweight='bold')
ax2.set_title('Distribuția Total Turiști per Categorie', fontsize=12, fontweight='bold', pad=20)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

# Participare rate
ax3 = axes[1, 0]
data_for_box2 = []
for cat in [0, 1, 2]:
    data_for_box2.append(features_df_active[features_df_active['category']==cat]['participation_rate'].values)
bp2 = ax3.boxplot(data_for_box2, labels=labels, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_ylabel('Rata de participare', fontsize=10, fontweight='bold')
ax3.set_title('Consistență Temporală (Rata de Participare)', fontsize=12, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, axis='y')

# Scatter: Total vs Recent activity
ax4 = axes[1, 1]
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = features_df_active['category'] == i
    ax4.scatter(features_df_active.loc[mask, 'total_tourists'], 
                features_df_active.loc[mask, 'recent_avg'],
                c=color, label=label, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax4.set_xlabel('Total turiști (1992-2024)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Medie recent (2020-2024)', fontsize=10, fontweight='bold')
ax4.set_title('Total Historic vs Activitate Recentă', fontsize=12, fontweight='bold', pad=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('output/13_caracteristici_categorii.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/13_caracteristici_categorii.png")
plt.close()

# ====================================================================================
# PLOT 3: Matrice confuzie și evaluare
# ====================================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_xlabel('Prezis', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10, fontweight='bold')
    ax.set_title(f'{name}\nAccuracy: {model.score(X_test, y_test):.3f}', 
                 fontsize=11, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('output/14_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/14_confusion_matrices.png")
plt.close()

# Generare raport
print("\n" + "=" * 80)
print("EVALUARE MODELE DE CLASIFICARE")
print("=" * 80)

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"\n{name}:")
    print(f"   Train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"   Test accuracy: {model.score(X_test, y_test):.3f}")

raport = f"""
================================================================================
RAPORT: DECISION BOUNDARIES
Clasificarea Tipurilor de Turiști și Vizualizarea Frontierelor de Decizie
================================================================================

1. OBIECTIV
   Separarea și clasificarea fluxurilor turistice pe baza caracteristicilor
   temporale și volumetrice, identificând decision boundaries între categorii.

2. CATEGORII DEFINITE
   - CIS: {(features_df_active['category'] == 0).sum()} țări
   - Europa: {(features_df_active['category'] == 1).sum()} țări
   - Altele: {(features_df_active['category'] == 2).sum()} țări

3. CARACTERISTICI UTILIZATE
   - Total turiști (1992-2024)
   - Medie anuală
   - Deviație standard
   - Activitate recentă (2020-2024)
   - Activitate timpurie (1992-1996)
   - Rata de participare

4. REZULTATE PCA
   - PC1 explică: {pca.explained_variance_ratio_[0]*100:.1f}% din varianță
   - PC2 explică: {pca.explained_variance_ratio_[1]*100:.1f}% din varianță
   - Total: {sum(pca.explained_variance_ratio_)*100:.1f}%

5. PERFORMANȚĂ MODELE
{chr(10).join([f'   {name}: Train {models[name].score(X_train, y_train):.3f}, Test {models[name].score(X_test, y_test):.3f}' for name in models.keys()])}

6. CONCLUZII
   - Separare clară între categorii CIS vs Non-CIS
   - Overlap între Europa și Altele (țări mici)
   - Decision boundaries complexe necesită modele non-liniare

7. FIȘIERE GENERATE
   - 12_decision_boundaries.png
   - 13_caracteristici_categorii.png
   - 14_confusion_matrices.png

================================================================================
"""

with open('output/raport_decision_boundaries.txt', 'w', encoding='utf-8') as f:
    f.write(raport)

print("\n✅ Raport salvat: output/raport_decision_boundaries.txt")
print("\n" + "=" * 80)
print("FINALIZAT: Decision Boundaries")
print("=" * 80)
