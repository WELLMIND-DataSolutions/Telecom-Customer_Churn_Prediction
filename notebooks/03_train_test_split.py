import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/EDA_final_dataset.csv")

print("📊 Dataset Shape:", df.shape)

# -----------------------------
# TARGET COLUMN
# -----------------------------
target = "churn"

# -----------------------------
# FEATURES & TARGET SPLIT
# -----------------------------
X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# CREATE DATAFRAMES
# -----------------------------
train_df = X_train.copy()
train_df[target] = y_train

test_df = X_test.copy()
test_df[target] = y_test

# -----------------------------
# SAVE FILES
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

# -----------------------------
# FINAL CHECK
# -----------------------------
print("\n✅ Train-Test Split Completed")
print("📦 Train Shape:", train_df.shape)
print("📦 Test Shape:", test_df.shape)
print("\n💾 Files Saved:")
print("   👉 data/processed/train.csv")
print("   👉 data/processed/test.csv")

# =========================================================
# 🎨 PROFESSIONAL WHITE THEME SETUP
# =========================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Custom white theme parameters
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#2c3e50',
    'axes.labelcolor': '#2c3e50',
    'text.color': '#2c3e50',
    'xtick.color': '#2c3e50',
    'ytick.color': '#2c3e50',
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'legend.facecolor': 'white',
    'legend.edgecolor': '#2c3e50',
    'legend.framealpha': 0.9,
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'figure.dpi': 300
})

# Create figures directory
os.makedirs('reports/figures', exist_ok=True)

print("="*70)
print("📊 GENERATING 3 MODERN TRAIN-TEST PLOTS (WHITE THEME)")
print("="*70)

# =========================================================
# PLOT 1: TARGET VARIABLE DISTRIBUTION
# =========================================================
print("\n📊 PLOT 1: Train-Test Distribution Comparison")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# Plot 1a: Target Distribution (Classification)
train_counts = y_train.value_counts(normalize=True) * 100
test_counts = y_test.value_counts(normalize=True) * 100

x = np.arange(len(train_counts))
width = 0.35

bars1 = axes[0].bar(x - width/2, train_counts.values, width, label='Train', 
                    color='#2E86AB', edgecolor='black', linewidth=1.5)
bars2 = axes[0].bar(x + width/2, test_counts.values, width, label='Test',
                    color='#A23B72', edgecolor='black', linewidth=1.5)

axes[0].set_xticks(x)
axes[0].set_xticklabels(['No Churn', 'Churn'] if len(train_counts) == 2 else train_counts.index, 
                        fontsize=11, fontweight='bold', color='#2c3e50')
axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold', color='#2c3e50')
axes[0].set_title(f'Target Distribution: Train vs Test\n{target.upper()}', 
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3, linestyle='--', axis='y', color='#cccccc')

# Add value labels on bars
for bar, val in zip(bars1, train_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#2c3e50')
for bar, val in zip(bars2, test_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#2c3e50')

# Plot 1b: Sample Sizes
sample_sizes = [len(X_train), len(X_test)]
sample_labels = ['Train', 'Test']
colors = ['#2E86AB', '#A23B72']

bars = axes[1].bar(sample_labels, sample_sizes, color=colors, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Number of Samples', fontsize=12, fontweight='bold', color='#2c3e50')
axes[1].set_title('Dataset Split Sizes', fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
axes[1].grid(True, alpha=0.3, linestyle='--', axis='y', color='#cccccc')

# Add value labels
for bar, size in zip(bars, sample_sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(sample_sizes)*0.02),
                f'{size:,}', ha='center', fontsize=12, fontweight='bold', color='#2c3e50')

# Add interpretation text
interpretation_text = f"""
📊 INTERPRETATION:
• Train Size: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)
• Test Size: {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)
• Target Distribution: {'Balanced' if abs(train_counts.iloc[0] - 50) < 15 else 'Imbalanced'}
"""
axes[1].text(0.02, 0.98, interpretation_text, transform=axes[1].transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#2c3e50', alpha=0.9),
            color='#2c3e50')

plt.suptitle('📊 TRAIN-TEST SPLIT ANALYSIS', fontsize=16, fontweight='bold', y=1.02, color='#2c3e50')
plt.tight_layout()
plt.savefig('reports/figures/plot1_train_test_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: reports/figures/plot1_train_test_distribution.png")
plt.close()

# =========================================================
# PLOT 2: FEATURE DISTRIBUTION OVERLAY (Top 6 Features)
# =========================================================
print("\n📊 PLOT 2: Feature Distribution Comparison")

# Select numeric columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 6:
    variances = X_train[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(6).index.tolist()
else:
    top_features = numeric_cols[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('white')
axes = axes.flatten()

for idx, feature in enumerate(top_features):
    ax = axes[idx]
    
    # Plot KDE for train and test
    sns.kdeplot(data=X_train[feature].dropna(), label='Train', 
                color='#2E86AB', linewidth=2.5, ax=ax)
    sns.kdeplot(data=X_test[feature].dropna(), label='Test',
                color='#A23B72', linewidth=2.5, linestyle='--', ax=ax)
    
    # Calculate KS statistic
    ks_stat, p_value = stats.ks_2samp(X_train[feature].dropna(), X_test[feature].dropna())
    
    # Styling
    ax.set_title(f'{feature}\nKS Test: p={p_value:.4f}', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Value', fontsize=10, color='#2c3e50')
    ax.set_ylabel('Density', fontsize=10, color='#2c3e50')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', color='#cccccc')
    ax.tick_params(colors='#2c3e50')
    ax.set_facecolor('white')
    
    # Add interpretation
    if p_value > 0.05:
        ax.text(0.98, 0.95, '✓ Similar Distribution', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8, edgecolor='green'),
                color='#155724')
    else:
        ax.text(0.98, 0.95, '⚠ Different Distribution',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#f8d7da', alpha=0.8, edgecolor='red'),
                color='#721c24')

# Remove empty subplot if any
for idx in range(len(top_features), 6):
    axes[idx].set_visible(False)

plt.suptitle('📈 FEATURE DISTRIBUTION: TRAIN vs TEST (KDE OVERLAY)', 
             fontsize=16, fontweight='bold', y=1.02, color='#2c3e50')
plt.tight_layout()
plt.savefig('reports/figures/plot2_feature_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: reports/figures/plot2_feature_distributions.png")
plt.close()

# =========================================================
# PLOT 3: CORRELATION HEATMAP COMPARISON
# =========================================================
print("\n📊 PLOT 3: Correlation Matrix Comparison")

# Select top numeric features for correlation
corr_features = top_features[:8] if len(top_features) >= 8 else top_features

# Calculate correlations
corr_train = X_train[corr_features].corr()
corr_test = X_test[corr_features].corr()
corr_diff = abs(corr_train - corr_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('white')

# Plot 3a: Train Correlation
sns.heatmap(corr_train, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=axes[0], cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            annot_kws={'size': 9, 'color': 'black'})
axes[0].set_title('TRAIN SET CORRELATIONS', fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
axes[0].tick_params(colors='#2c3e50')

# Plot 3b: Test Correlation
sns.heatmap(corr_test, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=axes[1], cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            annot_kws={'size': 9, 'color': 'black'})
axes[1].set_title('TEST SET CORRELATIONS', fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
axes[1].tick_params(colors='#2c3e50')

# Plot 3c: Correlation Difference
sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='YlOrRd', 
            square=True, ax=axes[2], cbar_kws={'shrink': 0.8, 'label': 'Absolute Difference'},
            annot_kws={'size': 9, 'color': 'black'})
axes[2].set_title('TRAIN-TEST DIFFERENCE\n(Higher = Less Consistent)', 
                  fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
axes[2].tick_params(colors='#2c3e50')

# Add interpretation text
fig.text(0.5, 0.02, 
        "📖 INTERPRETATION: Small differences in correlation matrices indicate good train-test split consistency.\n"
        "Red/Blue patterns should look similar between Train and Test. Yellow/Orange cells show where relationships differ.",
        fontsize=10, color='#2c3e50', ha='center',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#2c3e50', alpha=0.9))

plt.suptitle('🔗 CORRELATION STRUCTURE: TRAIN vs TEST', 
             fontsize=16, fontweight='bold', y=1.02, color='#2c3e50')
plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig('reports/figures/plot3_correlation_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: reports/figures/plot3_correlation_comparison.png")
plt.close()

# =========================================================
# SUMMARY REPORT WITH SUPPORTIVE DETAILS
# =========================================================
print("\n" + "="*70)
print("✅ 3 MODERN TRAIN-TEST PLOTS GENERATED SUCCESSFULLY!")
print("="*70)

print("\n📁 SAVE LOCATION: reports/figures/")
print("\n📊 PLOTS GENERATED:")
print("   ┌─────────────────────────────────────────────────────────────────────┐")
print("   │ 1️⃣ plot1_train_test_distribution.png                               │")
print("   │    → Target distribution comparison (Bar Chart)                     │")
print("   │    → Dataset split sizes with counts                                │")
print("   │                                                                     │")
print("   │ 2️⃣ plot2_feature_distributions.png                                 │")
print("   │    → KDE overlay plots for top 6 features                          │")
print("   │    → KS Test p-values for statistical validation                   │")
print("   │    → Green/Red indicators for distribution similarity              │")
print("   │                                                                     │")
print("   │ 3️⃣ plot3_correlation_comparison.png                                │")
print("   │    → Train set correlation matrix                                  │")
print("   │    → Test set correlation matrix                                   │")
print("   │    → Difference heatmap showing inconsistencies                    │")
print("   └─────────────────────────────────────────────────────────────────────┘")

print("\n🎨 DESIGN FEATURES:")
print("   • ✓ Clean white professional background")
print("   • ✓ High resolution (300 DPI PNG)")
print("   • ✓ Dark text for readability")
print("   • ✓ Aesthetic color combinations (Blue/Pink theme)")
print("   • ✓ Clear legends and labels")
print("   • ✓ Statistical validation (KS Test)")
print("   • ✓ Professional grid styling")
print("   • ✓ Print-friendly (saves ink/toner)")

print("\n📖 SUPPORTIVE DETAILS & INTERPRETATION:")
print("   ┌─────────────────────────────────────────────────────────────────────┐")
print("   │ PLOT 1 - DISTRIBUTION CHECK:                                        │")
print("   │   • Check if target percentage is similar in train/test            │")
print("   │   • Ideal: ±2% difference between splits                           │")
print("   │                                                                     │")
print("   │ PLOT 2 - FEATURE CONSISTENCY:                                       │")
print("   │   • Green ✓ → p > 0.05 (Good - similar distribution)               │")
print("   │   • Red ⚠ → p < 0.05 (Warning - different distribution)            │")
print("   │   • Action: Investigate features with red warnings                 │")
print("   │                                                                     │")
print("   │ PLOT 3 - CORRELATION STABILITY:                                     │")
print("   │   • Train and Test matrices should look similar                    │")
print("   │   • Yellow/Orange cells show relationship differences              │")
print("   │   • Large differences → Unstable feature relationships             │")
print("   └─────────────────────────────────────────────────────────────────────┘")

# Print feature distribution summary
print("\n⚠️ FEATURE DISTRIBUTION CHECK (KS TEST RESULTS):")
print("   " + "-"*50)
problem_features = []
for feature in top_features:
    ks_stat, p_value = stats.ks_2samp(X_train[feature].dropna(), X_test[feature].dropna())
    if p_value < 0.05:
        problem_features.append(feature)
        print(f"   ⚠ {feature:25s} → p={p_value:.6f} → DISTRIBUTION MISMATCH")
    else:
        print(f"   ✓ {feature:25s} → p={p_value:.6f} → Good distribution")

if len(problem_features) > 0:
    print(f"\n   ⚠️ WARNING: {len(problem_features)} feature(s) have different distributions!")
    print(f"   → Consider stratified sampling or feature engineering for: {problem_features}")
else:
    print(f"\n   ✅ PERFECT! All features have similar distributions between train and test!")

print("\n" + "="*70)
print("🎉 All plots are ready! Check the 'reports/figures/' folder")
print("="*70)

# Display file sizes
print("\n📁 FILE SIZES:")
for plot_file in ['plot1_train_test_distribution.png', 'plot2_feature_distributions.png', 'plot3_correlation_comparison.png']:
    filepath = f'reports/figures/{plot_file}'
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"   • {plot_file}: {size_kb:.1f} KB")