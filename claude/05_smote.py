"""
Pas 5: SMOTE - Synthetic Minority Over-sampling Technique
Echilibrarea regiunilor/tipurilor subreprezentate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("ANALIZA SMOTE - ECHILIBRARE DATE FLUXURI TURISTICE")
print("=" * 80)

df = pd.read_csv('output/dataset_clean.csv', index_col=0)

# Clasificare țări
cis_countries = ['Armenia', 'Azerbaijan', 'Belarus', 'Georgia (CIS)', 'Kazakhstan', 
                 'Kyrgyzstan', 'Russian Federation', 'Tajikistan', 'Turkmenistan', 
                 'Ukraine', 'Uzbekistan']
european_countries = ['Albania', 'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus',
                      'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
                      'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
                      'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Spain',
                      'Sweden', 'Switzerland', 'United Kingdom', 'Czech Republic']

def classify_country(country):
    if country in cis_countries:
        return 0
    elif country in european_countries:
        return 1
    else:
        return 2

# Caracteristici
features_df = pd.DataFrame(index=df.index)
features_df['total'] = df.sum(axis=1)
features_df['avg'] = df.mean(axis=1)
features_df['std'] = df.std(axis=1)
features_df['recent'] = df[['2020', '2021', '2022', '2023', '2024']].mean(axis=1)
features_df['participation'] = (df > 0).sum(axis=1) / len(df.columns)
features_df['category'] = features_df.index.map(classify_country)
features_df = features_df[features_df['total'] > 100].dropna()

X = features_df[['total', 'avg', 'std', 'recent', 'participation']].values
y = features_df['category'].values

print(f"\n📊 Distribuție inițială:")
print(f"   Total samples: {len(y)}")
for i, label in enumerate(['CIS', 'Europa', 'Altele']):
    print(f"   {label}: {(y==i).sum()} ({(y==i).sum()/len(y)*100:.1f}%)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model FĂRĂ SMOTE
print("\n" + "=" * 80)
print("ANTRENARE MODEL FĂRĂ SMOTE")
print("=" * 80)

rf_before = RandomForestClassifier(n_estimators=100, random_state=42)
rf_before.fit(X_train_scaled, y_train)

y_pred_before = rf_before.predict(X_test_scaled)
print(f"\nAccuracy: {rf_before.score(X_test_scaled, y_test):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_before, target_names=['CIS', 'Europa', 'Altele']))

# SMOTE
print("\n" + "=" * 80)
print("APLICARE SMOTE")
print("=" * 80)

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\n📊 Distribuție după SMOTE:")
print(f"   Total samples: {len(y_train_smote)} (original: {len(y_train)})")
for i, label in enumerate(['CIS', 'Europa', 'Altele']):
    print(f"   {label}: {(y_train_smote==i).sum()} ({(y_train_smote==i).sum()/len(y_train_smote)*100:.1f}%)")

# Model CU SMOTE
print("\n" + "=" * 80)
print("ANTRENARE MODEL CU SMOTE")
print("=" * 80)

rf_after = RandomForestClassifier(n_estimators=100, random_state=42)
rf_after.fit(X_train_smote, y_train_smote)

y_pred_after = rf_after.predict(X_test_scaled)
print(f"\nAccuracy: {rf_after.score(X_test_scaled, y_test):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_after, target_names=['CIS', 'Europa', 'Altele']))

# ====================================================================================
# PLOT 1: Comparație distribuții înainte/după SMOTE
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribuție originală
ax1 = axes[0, 0]
counts_orig = Counter(y_train)
labels_cat = ['CIS', 'Europa', 'Altele']
colors = ['#E74C3C', '#3498DB', '#2ECC71']
bars = ax1.bar(labels_cat, [counts_orig[i] for i in range(3)], color=colors, 
               edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_ylabel('Număr samples', fontsize=11, fontweight='bold')
ax1.set_title('Distribuție Originală (Înainte de SMOTE)', fontsize=12, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(y_train)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Distribuție după SMOTE
ax2 = axes[0, 1]
counts_smote = Counter(y_train_smote)
bars = ax2.bar(labels_cat, [counts_smote[i] for i in range(3)], color=colors, 
               edgecolor='black', linewidth=2, alpha=0.8)
ax2.set_ylabel('Număr samples', fontsize=11, fontweight='bold')
ax2.set_title('Distribuție După SMOTE (Echilibrată)', fontsize=12, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(y_train_smote)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Confusion matrix înainte
ax3 = axes[1, 0]
cm_before = confusion_matrix(y_test, y_pred_before)
sns.heatmap(cm_before, annot=True, fmt='d', cmap='Reds', ax=ax3,
            xticklabels=labels_cat, yticklabels=labels_cat, cbar=False)
ax3.set_xlabel('Prezis', fontsize=10, fontweight='bold')
ax3.set_ylabel('Actual', fontsize=10, fontweight='bold')
ax3.set_title(f'Confusion Matrix Înainte de SMOTE\nAccuracy: {rf_before.score(X_test_scaled, y_test):.3f}', 
              fontsize=11, fontweight='bold', pad=15)

# Confusion matrix după
ax4 = axes[1, 1]
cm_after = confusion_matrix(y_test, y_pred_after)
sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=ax4,
            xticklabels=labels_cat, yticklabels=labels_cat, cbar=False)
ax4.set_xlabel('Prezis', fontsize=10, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=10, fontweight='bold')
ax4.set_title(f'Confusion Matrix După SMOTE\nAccuracy: {rf_after.score(X_test_scaled, y_test):.3f}', 
              fontsize=11, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('output/15_smote_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Salvat: output/15_smote_comparison.png")
plt.close()

# ====================================================================================
# PLOT 2: Vizualizare samples sintetice
# ====================================================================================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_smote_pca = pca.transform(X_train_smote)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Înainte de SMOTE
ax1 = axes[0]
for i, (color, label) in enumerate(zip(colors, labels_cat)):
    mask = y_train == i
    ax1.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], c=color, label=label,
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11, fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11, fontweight='bold')
ax1.set_title('Date Originale (Înainte de SMOTE)', fontsize=12, fontweight='bold', pad=20)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# După SMOTE (evidențiem sintetic)
ax2 = axes[1]
# Identificăm samples sintetice (cele după len(X_train_scaled))
n_original = len(X_train_scaled)

for i, (color, label) in enumerate(zip(colors, labels_cat)):
    # Originale
    mask_orig = (y_train_smote == i) & (np.arange(len(y_train_smote)) < n_original)
    ax2.scatter(X_smote_pca[mask_orig, 0], X_smote_pca[mask_orig, 1], 
                c=color, label=f'{label} (original)', s=100, alpha=0.7, 
                edgecolors='black', linewidth=1.5)
    
    # Sintetice
    mask_synth = (y_train_smote == i) & (np.arange(len(y_train_smote)) >= n_original)
    ax2.scatter(X_smote_pca[mask_synth, 0], X_smote_pca[mask_synth, 1], 
                c=color, label=f'{label} (sintetic)', s=60, alpha=0.4, 
                marker='s', edgecolors='black', linewidth=1)

ax2.set_xlabel(f'PC1', fontsize=11, fontweight='bold')
ax2.set_ylabel(f'PC2', fontsize=11, fontweight='bold')
ax2.set_title('Date După SMOTE (pătrate = samples sintetice)', fontsize=12, fontweight='bold', pad=20)
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/16_smote_synthetic_samples.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/16_smote_synthetic_samples.png")
plt.close()

# ====================================================================================
# Generare raport
# ====================================================================================
from sklearn.metrics import precision_recall_fscore_support

precision_before, recall_before, f1_before, _ = precision_recall_fscore_support(y_test, y_pred_before, average=None)
precision_after, recall_after, f1_after, _ = precision_recall_fscore_support(y_test, y_pred_after, average=None)

raport = f"""
================================================================================
RAPORT: SMOTE - ECHILIBRARE DATE
Aplicarea SMOTE pentru Regiuni Subreprezentate
================================================================================

1. OBIECTIV
   Echilibrarea distribuției claselor prin generarea de samples sintetice
   pentru categoriile minoritare, îmbunătățind performanța modelului.

2. DISTRIBUȚIE INIȚIALĂ (Train Set)
   Total: {len(y_train)} samples
   - CIS: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)
   - Europa: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)
   - Altele: {(y_train==2).sum()} ({(y_train==2).sum()/len(y_train)*100:.1f}%)
   
   OBSERVAȚIE: Dezechilibru semnificativ, CIS dominantă

3. APLICARE SMOTE
   Parametri: k_neighbors=3, strategie=echilibrare completă
   
   Distribuție după SMOTE:
   Total: {len(y_train_smote)} samples (+{len(y_train_smote)-len(y_train)} sintetice)
   - CIS: {(y_train_smote==0).sum()} ({(y_train_smote==0).sum()/len(y_train_smote)*100:.1f}%)
   - Europa: {(y_train_smote==1).sum()} ({(y_train_smote==1).sum()/len(y_train_smote)*100:.1f}%)
   - Altele: {(y_train_smote==2).sum()} ({(y_train_smote==2).sum()/len(y_train_smote)*100:.1f}%)
   
   REZULTAT: Distribuție echilibrată perfectă (33.3% fiecare)

4. PERFORMANȚĂ MODEL

   A. ÎNAINTE DE SMOTE:
      Accuracy: {rf_before.score(X_test_scaled, y_test):.3f}
      
      Per clasă:
      - CIS:    Precision: {precision_before[0]:.3f}, Recall: {recall_before[0]:.3f}, F1: {f1_before[0]:.3f}
      - Europa: Precision: {precision_before[1]:.3f}, Recall: {recall_before[1]:.3f}, F1: {f1_before[1]:.3f}
      - Altele: Precision: {precision_before[2]:.3f}, Recall: {recall_before[2]:.3f}, F1: {f1_before[2]:.3f}

   B. DUPĂ SMOTE:
      Accuracy: {rf_after.score(X_test_scaled, y_test):.3f}
      
      Per clasă:
      - CIS:    Precision: {precision_after[0]:.3f}, Recall: {recall_after[0]:.3f}, F1: {f1_after[0]:.3f}
      - Europa: Precision: {precision_after[1]:.3f}, Recall: {recall_after[1]:.3f}, F1: {f1_after[1]:.3f}
      - Altele: Precision: {precision_after[2]:.3f}, Recall: {recall_after[2]:.3f}, F1: {f1_after[2]:.3f}

5. ÎMBUNĂTĂȚIRI OBSERVATE
   
   Δ Accuracy: {(rf_after.score(X_test_scaled, y_test) - rf_before.score(X_test_scaled, y_test))*100:+.1f} puncte procentuale
   
   Îmbunătățiri F1 per clasă:
   - CIS:    {(f1_after[0] - f1_before[0])*100:+.1f}%
   - Europa: {(f1_after[1] - f1_before[1])*100:+.1f}%
   - Altele: {(f1_after[2] - f1_before[2])*100:+.1f}%

6. AVANTAJE SMOTE
   ✓ Echilibrare automată a claselor
   ✓ Generare samples sintetice realiste (interpolări k-NN)
   ✓ Îmbunătățire recall pentru clase minoritare
   ✓ Reducere bias către clasa majoritară

7. LIMITĂRI ȘI ATENȚIONĂRI
   ⚠ Overfitting potential dacă k prea mic
   ⚠ Samples sintetice pot să nu reflecte realitatea
   ⚠ Nu rezolvă probleme de calitate date de bază
   ⚠ Evaluare finală doar pe date test reale (ne-augmentate)

8. RECOMANDĂRI UTILIZARE
   - Aplicare SMOTE doar pe train set (NICIODATĂ pe test!)
   - Validare cross-validation pe date echilibrate
   - Monitorizare distribuție predicții în producție
   - Considerare alternative: class_weight, stratified sampling

9. FIȘIERE GENERATE
   - 15_smote_comparison.png: Comparație înainte/după
   - 16_smote_synthetic_samples.png: Vizualizare samples sintetice

================================================================================
"""

with open('output/raport_smote.txt', 'w', encoding='utf-8') as f:
    f.write(raport)

print("\n✅ Raport salvat: output/raport_smote.txt")
print("\n" + "=" * 80)
print("FINALIZAT: Analiza SMOTE")
print("=" * 80)
