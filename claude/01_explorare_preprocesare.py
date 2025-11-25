"""
Pas 1: Explorare și Preprocesare Date
Analiza fluxului de turiști în Republica Moldova
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurare stil
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Încărcare date
print("=" * 80)
print("EXPLORARE ȘI PREPROCESARE DATE - FLUXURI TURISTICE MOLDOVA")
print("=" * 80)

df = pd.read_csv('../dataset_SAD.csv', index_col=0)
print(f"\n📊 Dimensiuni dataset: {df.shape}")
print(f"   - Țări: {df.shape[0]}")
print(f"   - Ani: {df.shape[1]} (1992-2024)")

# Informații generale
print("\n📋 Primele țări din dataset:")
print(df.head(10))

print("\n📈 Statistici descriptive generale:")
print(df.describe())

# Identificare valori lipsă
missing_values = df.isnull().sum().sum()
print(f"\n🔍 Valori lipsă: {missing_values}")

# Analiza evoluției totale
total_per_year = df.sum(axis=0)
print("\n📊 Total turiști pe an:")
print(total_per_year)

# Salvare rezultate
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# ====================================================================================
# PLOT 1: Evoluția totală a fluxului turistic (1992-2024)
# ====================================================================================
fig, ax = plt.subplots(figsize=(16, 6))
years = df.columns.astype(int)
ax.plot(years, total_per_year.values, marker='o', linewidth=2, markersize=6, color='#2E86AB')
ax.fill_between(years, total_per_year.values, alpha=0.3, color='#2E86AB')
ax.set_xlabel('An', fontsize=12, fontweight='bold')
ax.set_ylabel('Număr total turiști', fontsize=12, fontweight='bold')
ax.set_title('Evoluția Totală a Fluxului Turistic în Moldova (1992-2024)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style='plain', axis='y')

# Adăugare annotări pentru evenimente importante
ax.axvline(x=2020, color='red', linestyle='--', alpha=0.7, label='COVID-19')
ax.axvline(x=2022, color='orange', linestyle='--', alpha=0.7, label='Război Ucraina')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('output/01_evolutie_totala.png', dpi=300, bbox_inches='tight')
print("\n✅ Salvat: output/01_evolutie_totala.png")
plt.close()

# ====================================================================================
# PLOT 2: Top 15 țări sursă de turiști (total cumulat 1992-2024)
# ====================================================================================
total_per_country = df.sum(axis=1).sort_values(ascending=False)
top15_countries = total_per_country.head(15)

fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("viridis", len(top15_countries))
bars = ax.barh(range(len(top15_countries)), top15_countries.values, color=colors)
ax.set_yticks(range(len(top15_countries)))
ax.set_yticklabels(top15_countries.index, fontsize=10)
ax.set_xlabel('Număr total turiști (1992-2024)', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Țări Sursă de Turiști în Moldova', fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()

# Adăugare valori pe bare
for i, (bar, value) in enumerate(zip(bars, top15_countries.values)):
    ax.text(value + max(top15_countries.values)*0.01, i, f'{int(value):,}', 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('output/02_top15_tari.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/02_top15_tari.png")
plt.close()

# ====================================================================================
# PLOT 3: Heatmap - Intensitatea fluxului turistic pe țări și perioade
# ====================================================================================
# Selectăm top 20 țări pentru vizibilitate
top20_countries = df.sum(axis=1).sort_values(ascending=False).head(20).index
df_top20 = df.loc[top20_countries]

# Grupare pe perioade de 5 ani
period_labels = []
period_data = []
for start_year in range(1992, 2024, 5):
    end_year = min(start_year + 4, 2024)
    period_cols = [str(y) for y in range(start_year, end_year + 1) if str(y) in df.columns]
    period_sum = df_top20[period_cols].sum(axis=1)
    period_data.append(period_sum)
    period_labels.append(f'{start_year}-{end_year}')

df_periods = pd.DataFrame(period_data, index=period_labels).T

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df_periods, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Număr turiști'}, 
            linewidths=0.5, ax=ax)
ax.set_xlabel('Perioadă', fontsize=12, fontweight='bold')
ax.set_ylabel('Țară', fontsize=12, fontweight='bold')
ax.set_title('Heatmap: Intensitatea Fluxului Turistic pe Țări și Perioade', 
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/03_heatmap_perioade.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/03_heatmap_perioade.png")
plt.close()

# ====================================================================================
# PLOT 4: Distribuția statistică a fluxurilor turistice
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribuția valorilor (log scale pentru vizibilitate)
ax1 = axes[0, 0]
all_values = df.values.flatten()
all_values_nonzero = all_values[all_values > 0]
ax1.hist(np.log10(all_values_nonzero), bins=50, color='#A23E48', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Log10(Număr turiști)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Frecvență', fontsize=10, fontweight='bold')
ax1.set_title('Distribuția Fluxurilor Turistice (Scală Logaritmică)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Box plot pe perioade
ax2 = axes[0, 1]
period_data_list = []
for start_year in range(1992, 2024, 8):
    end_year = min(start_year + 7, 2024)
    period_cols = [str(y) for y in range(start_year, end_year + 1) if str(y) in df.columns]
    period_values = df[period_cols].values.flatten()
    period_data_list.append(period_values[period_values > 0])

bp = ax2.boxplot(period_data_list, labels=['1992-99', '2000-07', '2008-15', '2016-23'], 
                 patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], sns.color_palette("Set2", 4)):
    patch.set_facecolor(color)
ax2.set_ylabel('Număr turiști', fontsize=10, fontweight='bold')
ax2.set_title('Distribuția pe Perioade (fără outlieri)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Categorii de țări: CIS vs Non-CIS
ax3 = axes[1, 0]
cis_countries = ['Armenia', 'Azerbaijan', 'Belarus', 'Georgia (CIS)', 'Kazakhstan', 
                 'Kyrgyzstan', 'Russian Federation', 'Tajikistan', 'Turkmenistan', 
                 'Ukraine', 'Uzbekistan']
cis_mask = df.index.isin(cis_countries)
cis_total = df[cis_mask].sum(axis=1).sum()
non_cis_total = df[~cis_mask].sum(axis=1).sum()

colors_pie = ['#FF6B6B', '#4ECDC4']
ax3.pie([cis_total, non_cis_total], labels=['Țări CIS', 'Țări Non-CIS'], 
        autopct='%1.1f%%', startangle=90, colors=colors_pie, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax3.set_title('Proporția Turiști: CIS vs Non-CIS', fontsize=11, fontweight='bold')

# Evoluția CIS vs Non-CIS în timp
ax4 = axes[1, 1]
cis_yearly = df[cis_mask].sum(axis=0)
non_cis_yearly = df[~cis_mask].sum(axis=0)
years = df.columns.astype(int)

ax4.plot(years, cis_yearly.values, marker='o', label='CIS', linewidth=2, color='#FF6B6B')
ax4.plot(years, non_cis_yearly.values, marker='s', label='Non-CIS', linewidth=2, color='#4ECDC4')
ax4.set_xlabel('An', fontsize=10, fontweight='bold')
ax4.set_ylabel('Număr turiști', fontsize=10, fontweight='bold')
ax4.set_title('Evoluția Comparativă: CIS vs Non-CIS', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/04_distributii_statistice.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/04_distributii_statistice.png")
plt.close()

# ====================================================================================
# PLOT 5: Sezonalitate și tendințe pe regiuni
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Evoluția top 5 țări
top5_countries = df.sum(axis=1).sort_values(ascending=False).head(5).index
ax1 = axes[0, 0]
for country in top5_countries:
    ax1.plot(years, df.loc[country].values, marker='o', label=country, linewidth=2, markersize=4)
ax1.set_xlabel('An', fontsize=10, fontweight='bold')
ax1.set_ylabel('Număr turiști', fontsize=10, fontweight='bold')
ax1.set_title('Evoluția Top 5 Țări Sursă', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Rata de creștere anuală (procent)
ax2 = axes[0, 1]
growth_rate = total_per_year.pct_change() * 100
ax2.bar(years[1:], growth_rate.values[1:], color=['green' if x > 0 else 'red' for x in growth_rate.values[1:]], 
        alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('An', fontsize=10, fontweight='bold')
ax2.set_ylabel('Rata de creștere (%)', fontsize=10, fontweight='bold')
ax2.set_title('Rata de Creștere Anuală a Fluxului Turistic', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Concentrare: Indice Herfindahl-Hirschman pe ani
ax3 = axes[1, 0]
hhi_values = []
for year in df.columns:
    year_data = df[year]
    total = year_data.sum()
    if total > 0:
        shares = (year_data / total) ** 2
        hhi = shares.sum() * 10000  # Multiplicat cu 10000 pentru standardizare
        hhi_values.append(hhi)
    else:
        hhi_values.append(0)

ax3.plot(years, hhi_values, marker='o', linewidth=2, color='#9B59B6', markersize=5)
ax3.fill_between(years, hhi_values, alpha=0.3, color='#9B59B6')
ax3.set_xlabel('An', fontsize=10, fontweight='bold')
ax3.set_ylabel('Indice HHI', fontsize=10, fontweight='bold')
ax3.set_title('Concentrarea Pieței Turistice (Indice Herfindahl-Hirschman)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=1500, color='orange', linestyle='--', alpha=0.7, label='Prag concentrare moderată')
ax3.axhline(y=2500, color='red', linestyle='--', alpha=0.7, label='Prag concentrare ridicată')
ax3.legend(fontsize=8)

# Top 10 țări în 2019 (pre-COVID) vs 2024 (recent)
ax4 = axes[1, 1]
top_2019 = df['2019'].sort_values(ascending=False).head(10)
top_2024 = df['2024'].sort_values(ascending=False).head(10)

# Creăm un set comun de țări
all_top_countries = list(set(top_2019.index) | set(top_2024.index))
x_pos = np.arange(len(all_top_countries))
width = 0.35

values_2019 = [df.loc[country, '2019'] if country in df.index else 0 for country in all_top_countries]
values_2024 = [df.loc[country, '2024'] if country in df.index else 0 for country in all_top_countries]

bars1 = ax4.barh(x_pos - width/2, values_2019, width, label='2019', color='#3498db', alpha=0.8)
bars2 = ax4.barh(x_pos + width/2, values_2024, width, label='2024', color='#e74c3c', alpha=0.8)

ax4.set_yticks(x_pos)
ax4.set_yticklabels(all_top_countries, fontsize=8)
ax4.set_xlabel('Număr turiști', fontsize=10, fontweight='bold')
ax4.set_title('Comparație Top Țări: 2019 vs 2024', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('output/05_tendinte_sezonalitate.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/05_tendinte_sezonalitate.png")
plt.close()

# ====================================================================================
# Preprocesare date pentru analize ulterioare
# ====================================================================================
print("\n" + "=" * 80)
print("PREPROCESARE DATE")
print("=" * 80)

# Salvare date curate
df_clean = df.fillna(0)  # Înlocuire valori lipsă cu 0 (absența turiștilor)
df_clean.to_csv('output/dataset_clean.csv')
print("\n✅ Dataset curat salvat: output/dataset_clean.csv")

# Statistici pe categorii
print("\n📊 Statistici pe categorii de țări:")
print(f"   - Țări CIS: {cis_mask.sum()} țări, {cis_total:,} turiști total")
print(f"   - Țări Non-CIS: {(~cis_mask).sum()} țări, {non_cis_total:,} turiști total")

print(f"\n📈 Tendințe principale:")
print(f"   - An cu cel mai mare flux: {total_per_year.idxmax()} ({int(total_per_year.max()):,} turiști)")
print(f"   - An cu cel mai mic flux: {total_per_year.idxmin()} ({int(total_per_year.min()):,} turiști)")
print(f"   - Medie anuală: {int(total_per_year.mean()):,} turiști")
print(f"   - Scădere 2020 (COVID-19): {((total_per_year['2020'] - total_per_year['2019']) / total_per_year['2019'] * 100):.1f}%")

# Generare raport text
raport = f"""
================================================================================
RAPORT: EXPLORARE ȘI PREPROCESARE DATE
Analiza Fluxului de Turiști în Republica Moldova (1992-2024)
================================================================================

1. DESCRIERE DATASET
   - Număr țări: {df.shape[0]}
   - Ani acoperire: {df.shape[1]} (1992-2024)
   - Total turiști (1992-2024): {int(df.sum().sum()):,}
   - Valori lipsă: {missing_values}

2. ȚĂRI SURSĂ PRINCIPALE
   Top 5 țări (total cumulat):
   {chr(10).join([f'   - {country}: {int(total_per_country[country]):,} turiști' for country in total_per_country.head(5).index])}

3. CATEGORII GEOGRAFICE
   - Țări CIS: {cis_mask.sum()} țări
     Total turiști CIS: {int(cis_total):,} ({cis_total/(cis_total+non_cis_total)*100:.1f}%)
   
   - Țări Non-CIS: {(~cis_mask).sum()} țări
     Total turiști Non-CIS: {int(non_cis_total):,} ({non_cis_total/(cis_total+non_cis_total)*100:.1f}%)

4. EVOLUȚIE TEMPORALĂ
   - An cu flux maxim: {total_per_year.idxmax()} ({int(total_per_year.max()):,} turiști)
   - An cu flux minim: {total_per_year.idxmin()} ({int(total_per_year.min()):,} turiști)
   - Medie anuală: {int(total_per_year.mean()):,} turiști
   - Deviație standard: {int(total_per_year.std()):,}

5. EVENIMENTE MAJORE
   - Scădere COVID-19 (2020): {((total_per_year['2020'] - total_per_year['2019']) / total_per_year['2019'] * 100):.1f}%
   - Scădere război Ucraina (2022): {((total_per_year['2022'] - total_per_year['2019']) / total_per_year['2019'] * 100):.1f}%
   - Recuperare 2024: {((total_per_year['2024'] - total_per_year['2020']) / total_per_year['2020'] * 100):.1f}%

6. CONCENTRARE PIAȚĂ
   - HHI mediu (1992-2024): {np.mean(hhi_values):.0f}
   - HHI 2024: {hhi_values[-1]:.0f}
   - Interpretare: {'Concentrare ridicată' if hhi_values[-1] > 2500 else 'Concentrare moderată' if hhi_values[-1] > 1500 else 'Piață diversificată'}

7. TENDINȚE OBSERVATE
   - Dominanța țărilor CIS în perioada 1992-2000
   - Diversificarea surselor după 2000
   - Creștere țări Non-CIS după 2010
   - Impact sever COVID-19 în 2020
   - Recuperare graduală post-pandemie
   - Volatilitate ridicată în anii 2020-2024

8. OBSERVAȚII PENTRU ANALIZE ULTERIOARE
   - Dataset conține valori 0 (absența fluxurilor) - nu sunt valori lipsă reale
   - Distribuție asimetrică: multe țări cu fluxuri mici, puține cu fluxuri mari
   - Sezonalitate anuală și tendințe pe termen lung sunt evidente
   - Necesită normalizare pentru analize comparative
   - Potențial bias către țări vecine și CIS

9. FIȘIERE GENERATE
   - 01_evolutie_totala.png: Evoluția anuală totală
   - 02_top15_tari.png: Top 15 țări sursă
   - 03_heatmap_perioade.png: Intensitate pe perioade
   - 04_distributii_statistice.png: Analiză statistică
   - 05_tendinte_sezonalitate.png: Tendințe și comparații
   - dataset_clean.csv: Dataset preprocesar

================================================================================
"""

with open('output/raport_explorare.txt', 'w', encoding='utf-8') as f:
    f.write(raport)

print("\n✅ Raport salvat: output/raport_explorare.txt")
print("\n" + "=" * 80)
print("FINALIZAT: Explorare și preprocesare date")
print("=" * 80)
