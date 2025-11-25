"""
Pas 2: Analiza Bias
Identificarea bias-ului geografic și testarea echității în fluxurile turistice
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configurare stil
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("ANALIZA BIAS - FLUXURI TURISTICE MOLDOVA")
print("=" * 80)

# Încărcare date
df = pd.read_csv('output/dataset_clean.csv', index_col=0)
print(f"\n📊 Dataset încărcat: {df.shape}")

# Identificare categorii geografice
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

asian_countries = ['China', 'Japan', 'India', 'South Korea', 'Israel', 'Turkey',
                   'United Arab Emirates', 'Pakistan', 'Afghanistan', 'Bangladesh',
                   'Iran', 'Iraq', 'Jordan', 'Lebanon', 'Syria', 'Thailand', 'Vietnam',
                   'Indonesia', 'Malaysia', 'Philippines', 'Singapore', 'Sri Lanka',
                   'Saudi Arabia', 'Kuwait', 'Qatar', 'Bahrain', 'Oman', 'Yemen']

american_countries = ['United States', 'Canada', 'Brazil', 'Argentina', 'Mexico',
                      'Chile', 'Colombia', 'Peru', 'Venezuela', 'Cuba', 'Ecuador']

# Clasificare țări
def classify_country(country):
    if country in cis_countries:
        return 'CIS'
    elif country in european_countries:
        return 'Europa'
    elif country in asian_countries:
        return 'Asia'
    elif country in american_countries:
        return 'America'
    elif 'Other countries' in country:
        return 'Altele'
    else:
        # Încercăm să clasificăm după continent
        if any(x in country for x in ['Africa', 'African']):
            return 'Africa'
        return 'Altele'

df_regions = pd.DataFrame(index=df.index)
df_regions['Total'] = df.sum(axis=1)
df_regions['Region'] = df_regions.index.map(classify_country)

print("\n📊 Distribuție pe regiuni:")
region_stats = df_regions.groupby('Region')['Total'].agg(['count', 'sum', 'mean', 'std'])
print(region_stats)

# ====================================================================================
# PLOT 1: Bias geografic - Distribuția pe regiuni
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Număr țări per regiune
ax1 = axes[0, 0]
region_counts = df_regions['Region'].value_counts()
colors1 = sns.color_palette("Set2", len(region_counts))
wedges, texts, autotexts = ax1.pie(region_counts.values, labels=region_counts.index, 
                                     autopct='%1.1f%%', startangle=90, colors=colors1,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax1.set_title('Distribuția Numărului de Țări pe Regiuni', fontsize=12, fontweight='bold', pad=20)

# Total turiști per regiune
ax2 = axes[0, 1]
region_totals = df_regions.groupby('Region')['Total'].sum().sort_values(ascending=False)
colors2 = sns.color_palette("viridis", len(region_totals))
bars = ax2.barh(range(len(region_totals)), region_totals.values, color=colors2, edgecolor='black')
ax2.set_yticks(range(len(region_totals)))
ax2.set_yticklabels(region_totals.index, fontsize=10)
ax2.set_xlabel('Număr total turiști (1992-2024)', fontsize=11, fontweight='bold')
ax2.set_title('Total Turiști pe Regiuni - BIAS IDENTIFICAT', fontsize=12, fontweight='bold', pad=20)
ax2.invert_yaxis()

for i, (bar, value) in enumerate(zip(bars, region_totals.values)):
    ax2.text(value + max(region_totals.values)*0.02, i, f'{int(value):,}', 
            va='center', fontsize=9, fontweight='bold')

# Gini coefficient pentru inegalitate
ax3 = axes[1, 0]
total_per_country = df.sum(axis=1).sort_values()
cumulative_tourists = np.cumsum(total_per_country.values)
cumulative_tourists_norm = cumulative_tourists / cumulative_tourists[-1]
cumulative_countries_norm = np.arange(1, len(total_per_country) + 1) / len(total_per_country)

# Calculare Gini
area_under_curve = np.trapz(cumulative_tourists_norm, cumulative_countries_norm)
gini = 1 - 2 * area_under_curve

ax3.plot(cumulative_countries_norm, cumulative_tourists_norm, linewidth=2.5, 
         color='#E74C3C', label='Curba Lorenz')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Linie egalitate perfectă', alpha=0.7)
ax3.fill_between(cumulative_countries_norm, cumulative_tourists_norm, 
                  alpha=0.3, color='#E74C3C')
ax3.set_xlabel('Proporție cumulativă țări', fontsize=11, fontweight='bold')
ax3.set_ylabel('Proporție cumulativă turiști', fontsize=11, fontweight='bold')
ax3.set_title(f'Curba Lorenz - Inegalitate Distribuție\nCoeficient Gini: {gini:.3f}', 
              fontsize=12, fontweight='bold', pad=20)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.text(0.5, 0.2, f'Gini = {gini:.3f}\n(0 = egalitate perfectă\n1 = inegalitate maximă)', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         ha='center')

# Top 20 vs Bottom 20 țări
ax4 = axes[1, 1]
top20 = total_per_country.tail(20)
bottom20 = total_per_country.head(20)

x = np.arange(20)
width = 0.35

bars1 = ax4.barh(x - width/2, top20.values[::-1], width, label='Top 20', 
                 color='#2ECC71', alpha=0.8, edgecolor='black')
bars2 = ax4.barh(x + width/2, bottom20.values[::-1], width, label='Bottom 20', 
                 color='#95A5A6', alpha=0.8, edgecolor='black')

ax4.set_yticks(x)
ax4.set_yticklabels(top20.index[::-1], fontsize=8)
ax4.set_xlabel('Număr total turiști', fontsize=11, fontweight='bold')
ax4.set_title('Bias: Top 20 vs Bottom 20 Țări', fontsize=12, fontweight='bold', pad=20)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('output/06_bias_geografic.png', dpi=300, bbox_inches='tight')
print("\n✅ Salvat: output/06_bias_geografic.png")
plt.close()

# ====================================================================================
# PLOT 2: Evoluția bias-ului în timp
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Evoluție proporție regiuni
ax1 = axes[0, 0]
years = df.columns.astype(int)
region_evolution = {}

for region in df_regions['Region'].unique():
    countries_in_region = df_regions[df_regions['Region'] == region].index
    region_yearly = df.loc[countries_in_region].sum(axis=0)
    region_evolution[region] = region_yearly.values

region_df = pd.DataFrame(region_evolution, index=years)
region_df_pct = region_df.div(region_df.sum(axis=1), axis=0) * 100

region_df_pct.plot(kind='area', stacked=True, ax=ax1, alpha=0.7, 
                    color=sns.color_palette("Set2", len(region_df_pct.columns)))
ax1.set_xlabel('An', fontsize=11, fontweight='bold')
ax1.set_ylabel('Proporție (%)', fontsize=11, fontweight='bold')
ax1.set_title('Evoluția Proporției Regiunilor în Timp', fontsize=12, fontweight='bold', pad=20)
ax1.legend(title='Regiune', fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
ax1.grid(True, alpha=0.3, axis='y')

# Concentrare: Top 5 țări
ax2 = axes[0, 1]
top5_share = []
for year in df.columns:
    year_total = df[year].sum()
    if year_total > 0:
        top5_year = df[year].nlargest(5).sum()
        share = (top5_year / year_total) * 100
        top5_share.append(share)
    else:
        top5_share.append(0)

ax2.plot(years, top5_share, marker='o', linewidth=2.5, markersize=6, 
         color='#E67E22', label='Top 5 țări')
ax2.fill_between(years, top5_share, alpha=0.3, color='#E67E22')
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Prag 50%')
ax2.set_xlabel('An', fontsize=11, fontweight='bold')
ax2.set_ylabel('Proporție top 5 țări (%)', fontsize=11, fontweight='bold')
ax2.set_title('Concentrarea Fluxurilor: Ponderea Top 5 Țări', fontsize=12, fontweight='bold', pad=20)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Shannon entropy (diversitate)
ax3 = axes[1, 0]
entropy_values = []
for year in df.columns:
    year_data = df[year]
    year_data_nonzero = year_data[year_data > 0]
    if len(year_data_nonzero) > 0:
        probabilities = year_data_nonzero / year_data_nonzero.sum()
        entropy = -np.sum(probabilities * np.log(probabilities))
        entropy_values.append(entropy)
    else:
        entropy_values.append(0)

ax3.plot(years, entropy_values, marker='s', linewidth=2.5, markersize=6, 
         color='#9B59B6', label='Shannon Entropy')
ax3.fill_between(years, entropy_values, alpha=0.3, color='#9B59B6')
ax3.set_xlabel('An', fontsize=11, fontweight='bold')
ax3.set_ylabel('Shannon Entropy', fontsize=11, fontweight='bold')
ax3.set_title('Diversitatea Surselor de Turiști (Shannon Entropy)', fontsize=12, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, 'Valori mai mari = diversitate mai mare\nValori mai mici = concentrare bias', 
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Test statistici Chi-square pentru bias geografic (ultimii 5 ani)
ax4 = axes[1, 1]
recent_years = ['2020', '2021', '2022', '2023', '2024']
recent_data = df[recent_years]

# Calculăm distribuția așteptată (uniformă) vs observată
# Agregăm datele pe regiuni pentru anii recenti
observed_by_region = {}
for region in df_regions['Region'].unique():
    countries_in_region = df_regions[df_regions['Region'] == region].index
    region_total = df.loc[countries_in_region, recent_years].sum().sum()
    observed_by_region[region] = region_total
observed_by_region = pd.Series(observed_by_region)
expected_uniform = np.full(len(observed_by_region), observed_by_region.sum() / len(observed_by_region))

chi2, p_value = stats.chisquare(observed_by_region.values, expected_uniform)

x_pos = np.arange(len(observed_by_region))
bars1 = ax4.bar(x_pos - 0.2, observed_by_region.values, 0.4, label='Observat', 
                color='#3498DB', alpha=0.8, edgecolor='black')
bars2 = ax4.bar(x_pos + 0.2, expected_uniform, 0.4, label='Așteptat (uniform)', 
                color='#E74C3C', alpha=0.8, edgecolor='black')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(observed_by_region.index, fontsize=9, rotation=45, ha='right')
ax4.set_ylabel('Număr turiști', fontsize=11, fontweight='bold')
ax4.set_title(f'Test Chi-Square: Bias vs Distribuție Uniformă (2020-2024)\nχ² = {chi2:.2f}, p = {p_value:.2e}', 
              fontsize=12, fontweight='bold', pad=20)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

if p_value < 0.001:
    conclusion = 'BIAS SEMNIFICATIV (p < 0.001)'
    color_text = 'red'
else:
    conclusion = 'Distribuție relativ echilibrată'
    color_text = 'green'

ax4.text(0.5, 0.95, conclusion, transform=ax4.transAxes, fontsize=11, 
         fontweight='bold', color=color_text, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('output/07_evolutie_bias.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/07_evolutie_bias.png")
plt.close()

# ====================================================================================
# PLOT 3: Analiza echității geografice
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Rata de participare pe regiuni (procent țări care trimit turiști)
ax1 = axes[0, 0]
participation_by_region = {}
for region in df_regions['Region'].unique():
    countries_in_region = df_regions[df_regions['Region'] == region].index
    participation_rates = []
    
    for year in df.columns:
        active_countries = (df.loc[countries_in_region, year] > 0).sum()
        total_countries = len(countries_in_region)
        if total_countries > 0:
            participation_rates.append((active_countries / total_countries) * 100)
        else:
            participation_rates.append(0)
    
    participation_by_region[region] = np.mean(participation_rates)

regions_sorted = sorted(participation_by_region.items(), key=lambda x: x[1], reverse=True)
regions_names = [x[0] for x in regions_sorted]
regions_values = [x[1] for x in regions_sorted]

colors = sns.color_palette("RdYlGn", len(regions_names))
bars = ax1.barh(range(len(regions_names)), regions_values, color=colors, edgecolor='black')
ax1.set_yticks(range(len(regions_names)))
ax1.set_yticklabels(regions_names, fontsize=10)
ax1.set_xlabel('Rata medie de participare (%)', fontsize=11, fontweight='bold')
ax1.set_title('Echitate: Rata Medie de Participare pe Regiuni', fontsize=12, fontweight='bold', pad=20)
ax1.invert_yaxis()

for i, (bar, value) in enumerate(zip(bars, regions_values)):
    ax1.text(value + 2, i, f'{value:.1f}%', va='center', fontsize=9, fontweight='bold')

# Distribuția intra-regională (variabilitate)
ax2 = axes[0, 1]
region_variability = {}
for region in df_regions['Region'].unique():
    countries_in_region = df_regions[df_regions['Region'] == region].index
    if len(countries_in_region) > 1:
        totals = df.loc[countries_in_region].sum(axis=1)
        cv = (totals.std() / totals.mean()) * 100 if totals.mean() > 0 else 0
        region_variability[region] = cv

regions_sorted_var = sorted(region_variability.items(), key=lambda x: x[1], reverse=True)
regions_names_var = [x[0] for x in regions_sorted_var]
regions_values_var = [x[1] for x in regions_sorted_var]

colors_var = sns.color_palette("YlOrRd", len(regions_names_var))
bars = ax2.barh(range(len(regions_names_var)), regions_values_var, 
                color=colors_var, edgecolor='black')
ax2.set_yticks(range(len(regions_names_var)))
ax2.set_yticklabels(regions_names_var, fontsize=10)
ax2.set_xlabel('Coeficient de variație (%)', fontsize=11, fontweight='bold')
ax2.set_title('Variabilitate Intra-Regională (Inegalitate în cadrul regiunilor)', 
              fontsize=12, fontweight='bold', pad=20)
ax2.invert_yaxis()

for i, (bar, value) in enumerate(zip(bars, regions_values_var)):
    ax2.text(value + max(regions_values_var)*0.02, i, f'{value:.1f}%', 
            va='center', fontsize=9, fontweight='bold')

# Comparație țări vecine vs non-vecine
ax3 = axes[1, 0]
neighbor_countries = ['Ukraine', 'Romania', 'Russian Federation', 'Belarus']
neighbor_data = df.loc[df.index.isin(neighbor_countries)].sum(axis=1)
non_neighbor_data = df.loc[~df.index.isin(neighbor_countries)].sum(axis=1)

neighbor_total = neighbor_data.sum()
non_neighbor_total = non_neighbor_data.sum()
neighbor_count = len(neighbor_data)
non_neighbor_count = len(non_neighbor_data)

categories = ['Țări vecine', 'Țări non-vecine']
totals = [neighbor_total, non_neighbor_total]
averages = [neighbor_total/neighbor_count, non_neighbor_total/non_neighbor_count]

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, totals, width, label='Total', 
                color='#3498DB', alpha=0.8, edgecolor='black')
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x_pos + width/2, averages, width, label='Medie per țară', 
                     color='#E74C3C', alpha=0.8, edgecolor='black')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(categories, fontsize=10)
ax3.set_ylabel('Total turiști', fontsize=11, fontweight='bold', color='#3498DB')
ax3_twin.set_ylabel('Medie per țară', fontsize=11, fontweight='bold', color='#E74C3C')
ax3.set_title('Bias Geografic: Țări Vecine vs Non-Vecine', fontsize=12, fontweight='bold', pad=20)
ax3.tick_params(axis='y', labelcolor='#3498DB')
ax3_twin.tick_params(axis='y', labelcolor='#E74C3C')

# Adăugare legenda
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

# Indici de echitate temporală (Theil index)
ax4 = axes[1, 1]
theil_values = []

for year in df.columns:
    year_data = df[year]
    year_data_nonzero = year_data[year_data > 0]
    
    if len(year_data_nonzero) > 1:
        n = len(year_data_nonzero)
        mean_val = year_data_nonzero.mean()
        theil = (1/n) * np.sum((year_data_nonzero / mean_val) * np.log(year_data_nonzero / mean_val))
        theil_values.append(theil)
    else:
        theil_values.append(0)

ax4.plot(years, theil_values, marker='D', linewidth=2.5, markersize=6, 
         color='#16A085', label='Theil Index')
ax4.fill_between(years, theil_values, alpha=0.3, color='#16A085')
ax4.set_xlabel('An', fontsize=11, fontweight='bold')
ax4.set_ylabel('Theil Index', fontsize=11, fontweight='bold')
ax4.set_title('Indice Theil - Măsură a Inegalității (Echitate Temporală)', 
              fontsize=12, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3)
ax4.text(0.05, 0.95, 'Valori mari = inegalitate mare\nValori mici = distribuție echitabilă', 
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('output/08_echitate_geografica.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/08_echitate_geografica.png")
plt.close()

# ====================================================================================
# Statistici finale și concluzii
# ====================================================================================
print("\n" + "=" * 80)
print("ANALIZĂ BIAS - REZULTATE")
print("=" * 80)

print(f"\n📊 COEFICIENT GINI: {gini:.3f}")
print(f"   Interpretare: {'Inegalitate severă' if gini > 0.5 else 'Inegalitate moderată' if gini > 0.3 else 'Distribuție relativ echitabilă'}")

print(f"\n📊 TEST CHI-SQUARE (2020-2024):")
print(f"   χ² = {chi2:.2f}")
print(f"   p-value = {p_value:.2e}")
print(f"   Concluzie: {'Bias geografic SEMNIFICATIV detectat' if p_value < 0.05 else 'Nu există bias semnificativ'}")

print(f"\n📊 CONCENTRARE TOP 5 ȚĂRI:")
print(f"   Medie 1992-2024: {np.mean(top5_share):.1f}%")
print(f"   2024: {top5_share[-1]:.1f}%")

print(f"\n📊 DIVERSITATE (Shannon Entropy):")
print(f"   Medie 1992-2024: {np.mean(entropy_values):.2f}")
print(f"   2024: {entropy_values[-1]:.2f}")
print(f"   Tendință: {'Diversificare' if entropy_values[-1] > np.mean(entropy_values) else 'Concentrare'}")

print(f"\n📊 BIAS ȚĂRI VECINE:")
print(f"   Total țări vecine (4 țări): {int(neighbor_total):,} turiști")
print(f"   Total țări non-vecine ({non_neighbor_count} țări): {int(non_neighbor_total):,} turiști")
print(f"   Medie per țară vecină: {int(neighbor_total/neighbor_count):,}")
print(f"   Medie per țară non-vecină: {int(non_neighbor_total/non_neighbor_count):,}")
print(f"   Ratio: {(neighbor_total/neighbor_count)/(non_neighbor_total/non_neighbor_count):.1f}x")

# Generare raport
raport = f"""
================================================================================
RAPORT: ANALIZA BIAS
Identificarea și Cuantificarea Bias-ului Geografic în Fluxurile Turistice
================================================================================

1. DEFINIREA BIAS-ULUI
   Bias-ul în contextul fluxurilor turistice se referă la favorizarea sistematică
   a anumitor regiuni geografice sau țări în detrimentul altora, rezultând o
   distribuție inegală și potențial inechitabilă a fluxurilor turistice.

2. INDICATORI PRINCIPALI DE BIAS

   A. COEFICIENT GINI: {gini:.3f}
      - Interval: 0 (egalitate perfectă) - 1 (inegalitate maximă)
      - Rezultat: {gini:.3f} indică {'INEGALITATE SEVERĂ' if gini > 0.5 else 'INEGALITATE MODERATĂ' if gini > 0.3 else 'DISTRIBUȚIE RELATIV ECHITABILĂ'}
      - Interpretare: {'Concentrarea extremă a fluxurilor pe câteva țări' if gini > 0.5 else 'Distribuție inegală notabilă' if gini > 0.3 else 'Distribuție acceptabilă'}

   B. TEST CHI-SQUARE (2020-2024)
      - χ² statistic: {chi2:.2f}
      - p-value: {p_value:.2e}
      - Concluzie: {'BIAS GEOGRAFIC SEMNIFICATIV DETECTAT (p < 0.05)' if p_value < 0.05 else 'Nu există dovezi statistice pentru bias'}
      - Semnificație: Distribuția observată diferă {'semnificativ' if p_value < 0.05 else 'nesemnificativ'} de o distribuție uniformă

   C. CONCENTRARE TOP 5 ȚĂRI
      - Medie istorică (1992-2024): {np.mean(top5_share):.1f}%
      - Valoare 2024: {top5_share[-1]:.1f}%
      - Tendință: {'Creștere concentrare (bias crescut)' if top5_share[-1] > np.mean(top5_share) else 'Scădere concentrare (diversificare)'}

   D. DIVERSITATE (Shannon Entropy)
      - Medie istorică: {np.mean(entropy_values):.2f}
      - Valoare 2024: {entropy_values[-1]:.2f}
      - Evoluție: {'Creștere diversitate (reducere bias)' if entropy_values[-1] > np.mean(entropy_values) else 'Scădere diversitate (creștere bias)'}

3. BIAS REGIONAL

   Distribuția pe regiuni:
{chr(10).join([f'   - {region}: {count} țări, {total:,} turiști (medie: {mean:.0f})' 
               for region, count, total, mean in zip(region_stats.index, region_stats['count'], 
                                                      region_stats['sum'], region_stats['mean'])])}

   Observații:
   - Dominanța regiunii: {region_stats['sum'].idxmax()} ({region_stats['sum'].max() / region_stats['sum'].sum() * 100:.1f}% din total)
   - Regiunea cu cea mai mică reprezentare: {region_stats['sum'].idxmin()}

4. BIAS GEOGRAFIC: ȚĂRI VECINE vs NON-VECINE

   Țări vecine (România, Ucraina, Rusia, Belarus):
   - Număr țări: {neighbor_count}
   - Total turiști: {int(neighbor_total):,}
   - Medie per țară: {int(neighbor_total/neighbor_count):,}

   Țări non-vecine:
   - Număr țări: {non_neighbor_count}
   - Total turiști: {int(non_neighbor_total):,}
   - Medie per țară: {int(non_neighbor_total/non_neighbor_count):,}

   Ratio medie vecine/non-vecine: {(neighbor_total/neighbor_count)/(non_neighbor_total/non_neighbor_count):.1f}x
   
   Concluzie: {'BIAS SEMNIFICATIV către țări vecine' if (neighbor_total/neighbor_count)/(non_neighbor_total/non_neighbor_count) > 10 else 'Bias moderat către țări vecine' if (neighbor_total/neighbor_count)/(non_neighbor_total/non_neighbor_count) > 5 else 'Bias redus'}

5. EVOLUȚIA BIAS-ULUI ÎN TIMP

   Perioada 1992-2000:
   - Concentrare CIS: {(region_df.loc[1992:2000, 'CIS'].sum() / region_df.loc[1992:2000].sum().sum() * 100):.1f}%
   - Diversitate redusă, dominanță fostelor state sovietice

   Perioada 2001-2010:
   - Diversificare graduală
   - Creștere ponderii europene

   Perioada 2011-2019:
   - Echilibrare relativă
   - Creștere țări non-CIS

   Perioada 2020-2024:
   - Impact COVID-19: colaps parțial al diversității
   - Recuperare 2023-2024: {(region_df.loc[2024].sum() / region_df.loc[2023].sum() - 1) * 100:.1f}% creștere

6. INDICATORI DE ECHITATE

   A. Rata de participare medie pe regiuni:
{chr(10).join([f'      - {region}: {rate:.1f}%' for region, rate in participation_by_region.items()])}

   B. Variabilitate intra-regională (coeficient de variație):
{chr(10).join([f'      - {region}: {cv:.1f}%' for region, cv in region_variability.items()])}

   C. Theil Index (măsură inegalitate):
      - Medie istorică: {np.mean(theil_values):.3f}
      - Valoare 2024: {theil_values[-1]:.3f}

7. IMPLICAȚII ȘI RECOMANDĂRI

   A. Probleme identificate:
      - Concentrare excesivă pe câteva țări sursă
      - Bias geografic semnificativ către țări vecine și CIS
      - Subreprezentare regiuni îndepărtate (America, Asia, Africa)
      - Volatilitate ridicată în perioade de criză

   B. Recomandări pentru reducerea bias-ului:
      1. Campanii de marketing țintite către regiuni subreprezentate
      2. Diversificarea rutelor de transport internațional
      3. Parteneriate cu agenții de turism din țări non-CIS
      4. Programe de facilitare vize pentru țări îndepărtate
      5. Promovare specifică în piețele asiatice și americane

   C. Pentru modelare predictivă:
      1. Utilizare tehnici de rebalansare (SMOTE, oversampling)
      2. Ponderare samples pentru compensarea bias-ului
      3. Stratificare pe regiuni în antrenare/validare
      4. Evaluare separată pe grupuri minoritare
      5. Metrici de fairness în plus față de acuratețea globală

8. CONCLUZII

   - Există un BIAS GEOGRAFIC SEMNIFICATIV în fluxurile turistice către Moldova
   - Coeficientul Gini ({gini:.3f}) indică inegalitate {'severă' if gini > 0.5 else 'moderată'}
   - Concentrarea pe top 5 țări este {'foarte ridicată' if np.mean(top5_share) > 70 else 'ridicată' if np.mean(top5_share) > 50 else 'moderată'} ({np.mean(top5_share):.1f}%)
   - Țările vecine și CIS domină fluxurile (bias proximitate geografică)
   - Tendință de diversificare observată post-2010, întreruptă de COVID-19
   - Necesită intervenții active pentru echilibrarea surselor turistice

9. FIȘIERE GENERATE
   - 06_bias_geografic.png: Distribuție regională și Curba Lorenz
   - 07_evolutie_bias.png: Evoluția temporală a bias-ului
   - 08_echitate_geografica.png: Analiză echitate și comparații

================================================================================
"""

with open('output/raport_bias.txt', 'w', encoding='utf-8') as f:
    f.write(raport)

print("\n✅ Raport salvat: output/raport_bias.txt")
print("\n" + "=" * 80)
print("FINALIZAT: Analiza Bias")
print("=" * 80)
