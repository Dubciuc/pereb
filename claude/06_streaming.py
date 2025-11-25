"""
Pas 6: Streaming - Procesare în Timp Real
Simularea procesării în timp real a fluxurilor turistice
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("ANALIZA STREAMING - PROCESARE TIMP REAL FLUXURI TURISTICE")
print("=" * 80)

df = pd.read_csv('output/dataset_clean.csv', index_col=0)
yearly_totals = df.sum(axis=0).values
years = df.columns.astype(int).values

print(f"\n📊 Simulare streaming pentru {len(years)} ani")

# Simulare streaming cu fereastră glisantă
window_size = 5
threshold_alert = 1.5  # Factor pentru alerte

streaming_data = []
alerts = []

print("\n" + "=" * 80)
print("PROCESARE STREAMING (Fereastră glisantă 5 ani)")
print("=" * 80)

for i in range(window_size, len(yearly_totals)):
    window = yearly_totals[i-window_size:i]
    current_value = yearly_totals[i]
    
    mean_window = np.mean(window)
    std_window = np.std(window)
    
    # Detectare anomalii
    z_score = abs((current_value - mean_window) / std_window) if std_window > 0 else 0
    
    # Trend
    trend = np.polyfit(range(window_size), window, 1)[0]
    
    streaming_data.append({
        'year': years[i],
        'value': current_value,
        'mean_window': mean_window,
        'std_window': std_window,
        'z_score': z_score,
        'trend': trend
    })
    
    if z_score > threshold_alert:
        alert_msg = f"🚨 ALERTĂ An {years[i]}: Anomalie detectată (Z-score: {z_score:.2f})"
        alerts.append({'year': years[i], 'z_score': z_score, 'value': current_value})
        print(alert_msg)

stream_df = pd.DataFrame(streaming_data)

print(f"\n📊 Total alerte: {len(alerts)}")

# ====================================================================================
# PLOT 1: Streaming cu fereastră glisantă
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Valori cu medie mobile
ax1 = axes[0, 0]
ax1.plot(stream_df['year'], stream_df['value'], 'o-', linewidth=2, 
         markersize=6, color='#3498DB', label='Valoare actuală')
ax1.plot(stream_df['year'], stream_df['mean_window'], 's--', linewidth=2, 
         markersize=5, color='#E74C3C', label=f'Medie glisantă ({window_size} ani)', alpha=0.7)
ax1.fill_between(stream_df['year'], 
                  stream_df['mean_window'] - threshold_alert*stream_df['std_window'],
                  stream_df['mean_window'] + threshold_alert*stream_df['std_window'],
                  alpha=0.2, color='orange', label='Prag alertă')

for alert in alerts:
    ax1.axvline(x=alert['year'], color='red', linestyle=':', alpha=0.5)
    ax1.scatter([alert['year']], [alert['value']], s=200, c='red', marker='X', 
                edgecolors='black', linewidth=2, zorder=5)

ax1.set_xlabel('An', fontsize=11, fontweight='bold')
ax1.set_ylabel('Număr turiști', fontsize=11, fontweight='bold')
ax1.set_title('Streaming cu Fereastră Glisantă și Detectare Anomalii', 
              fontsize=12, fontweight='bold', pad=20)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Z-scores
ax2 = axes[0, 1]
colors_z = ['red' if z > threshold_alert else 'green' for z in stream_df['z_score']]
ax2.bar(stream_df['year'], stream_df['z_score'], color=colors_z, edgecolor='black', alpha=0.7)
ax2.axhline(y=threshold_alert, color='red', linestyle='--', linewidth=2, 
            label=f'Prag alertă ({threshold_alert}σ)')
ax2.set_xlabel('An', fontsize=11, fontweight='bold')
ax2.set_ylabel('Z-Score', fontsize=11, fontweight='bold')
ax2.set_title('Detectare Anomalii (Z-Score)', fontsize=12, fontweight='bold', pad=20)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Trend
ax3 = axes[1, 0]
colors_trend = ['green' if t > 0 else 'red' for t in stream_df['trend']]
ax3.bar(stream_df['year'], stream_df['trend'], color=colors_trend, edgecolor='black', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('An', fontsize=11, fontweight='bold')
ax3.set_ylabel('Trend (panta)', fontsize=11, fontweight='bold')
ax3.set_title('Tendință Locală (Fereastră 5 ani)', fontsize=12, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, axis='y')

# Volatilitate (std glisant)
ax4 = axes[1, 1]
ax4.plot(stream_df['year'], stream_df['std_window'], 'o-', linewidth=2.5, 
         markersize=6, color='#9B59B6')
ax4.fill_between(stream_df['year'], stream_df['std_window'], alpha=0.3, color='#9B59B6')
ax4.set_xlabel('An', fontsize=11, fontweight='bold')
ax4.set_ylabel('Deviație standard (fereastră)', fontsize=11, fontweight='bold')
ax4.set_title('Volatilitate în Timp Real', fontsize=12, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/17_streaming_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Salvat: output/17_streaming_analysis.png")
plt.close()

# ====================================================================================
# PLOT 2: Top țări streaming
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

periods = [('2015-2019', ['2015', '2016', '2017', '2018', '2019']),
           ('2020-2024', ['2020', '2021', '2022', '2023', '2024'])]

for idx, (period_name, period_years) in enumerate(periods):
    ax = axes[idx, 0]
    period_data = df[period_years].sum(axis=1).sort_values(ascending=False).head(10)
    
    colors_bar = sns.color_palette("viridis", len(period_data))
    bars = ax.barh(range(len(period_data)), period_data.values, color=colors_bar, edgecolor='black')
    ax.set_yticks(range(len(period_data)))
    ax.set_yticklabels(period_data.index, fontsize=9)
    ax.set_xlabel('Număr turiști', fontsize=10, fontweight='bold')
    ax.set_title(f'Top 10 Țări: {period_name}', fontsize=11, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, value) in enumerate(zip(bars, period_data.values)):
        ax.text(value + max(period_data.values)*0.02, i, f'{int(value):,}', 
                va='center', fontsize=8)

# Comparație rate de creștere
ax = axes[0, 1]
growth_rates = []
for country in df.index[:20]:
    if df.loc[country, '2015':'2019'].sum() > 0:
        growth = ((df.loc[country, '2020':'2024'].sum() - df.loc[country, '2015':'2019'].sum()) / 
                  df.loc[country, '2015':'2019'].sum() * 100)
        growth_rates.append((country, growth))

growth_rates = sorted(growth_rates, key=lambda x: abs(x[1]), reverse=True)[:10]
countries_g, rates_g = zip(*growth_rates)

colors_growth = ['green' if r > 0 else 'red' for r in rates_g]
bars = ax.barh(range(len(rates_g)), rates_g, color=colors_growth, edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(countries_g)))
ax.set_yticklabels(countries_g, fontsize=9)
ax.set_xlabel('Rată creștere (%)', fontsize=10, fontweight='bold')
ax.set_title('Schimbări Majore: 2015-19 vs 2020-24', fontsize=11, fontweight='bold', pad=15)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Emergende: țări cu creștere recentă
ax = axes[1, 1]
recent_growth = df[['2023', '2024']].sum(axis=1) / (df[['2020', '2021', '2022']].sum(axis=1) + 1)
emerging = recent_growth.sort_values(ascending=False).head(15)

colors_em = sns.color_palette("RdYlGn", len(emerging))
bars = ax.barh(range(len(emerging)), emerging.values, color=colors_em, edgecolor='black')
ax.set_yticks(range(len(emerging)))
ax.set_yticklabels(emerging.index, fontsize=8)
ax.set_xlabel('Ratio 2023-24 / 2020-22', fontsize=10, fontweight='bold')
ax.set_title('Țări Emergente (Tendințe Recente)', fontsize=11, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('output/18_streaming_trends.png', dpi=300, bbox_inches='tight')
print("✅ Salvat: output/18_streaming_trends.png")
plt.close()

# Raport
raport = f"""
================================================================================
RAPORT: STREAMING - PROCESARE TIMP REAL
Monitorizare și Detectare Tendințe în Timp Real
================================================================================

1. OBIECTIV
   Simularea procesării streaming a fluxurilor turistice cu detectare
   automată a anomaliilor și tendințelor emergente.

2. METODOLOGIE
   - Fereastră glisantă: {window_size} ani
   - Prag alertă: {threshold_alert}σ (deviații standard)
   - Metrici: Z-score, trend local, volatilitate

3. ALERTE DETECTATE
   Total: {len(alerts)} anomalii

{chr(10).join([f'   - An {a["year"]}: Z-score {a["z_score"]:.2f}, Valoare {int(a["value"]):,}' for a in alerts])}

4. TENDINȚE IDENTIFICATE
   Perioada 2015-2019: Creștere stabilă
   Perioada 2020-2024: Volatilitate ridicată (COVID-19, război)
   
   Recuperare: {((yearly_totals[-1] - yearly_totals[-5]) / yearly_totals[-5] * 100):+.1f}% față de 2020

5. AVANTAJE STREAMING
   ✓ Detectare rapidă anomalii
   ✓ Adaptare în timp real
   ✓ Alerte automate
   ✓ Identificare tendințe emergente

6. APLICAȚII PRACTICE
   - Dashboard timp real pentru autorități turism
   - Sistem alertă pentru evenimente neașteptate
   - Recomandări adaptive campanii marketing
   - Optimizare capacități (transport, cazare)

7. FIȘIERE GENERATE
   - 17_streaming_analysis.png
   - 18_streaming_trends.png

================================================================================
"""

with open('output/raport_streaming.txt', 'w', encoding='utf-8') as f:
    f.write(raport)

print("\n✅ Raport salvat: output/raport_streaming.txt")
print("\n" + "=" * 80)
print("FINALIZAT: Analiza Streaming")
print("=" * 80)
