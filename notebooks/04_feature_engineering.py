import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import chi2_contingency

df = pd.read_csv("data/processed/train.csv")

print("Original Shape:", df.shape)
print("Missing Values:", df.isna().sum().sum())
print("Duplicates:", df.duplicated().sum())

# =========================================================
# 🔹 1. MISSING VALUES HANDLING (TELECOM LOGIC)
# =========================================================

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna("No_Activity")

# =========================================================
# 🔹 2. NUMERIC COLUMNS (FIXED: Remove future columns)
# =========================================================

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if 'churn' in num_cols:
    num_cols.remove('churn')

# 🔴 CRITICAL FIX 1: Remove any month 9 columns (future data leakage)
num_cols = [col for col in num_cols if '_9' not in col]
print(f"\n✅ Using {len(num_cols)} past-month columns only (no _9 columns)")

# =========================================================
# 🔥 3. BASIC BEHAVIOR FEATURES
# =========================================================

df["total_activity"] = df[num_cols].sum(axis=1)
df["avg_usage"] = df[num_cols].mean(axis=1)
# 🔴 FIX 2: Add fillna(0) for variance
df["usage_variability"] = df[num_cols].var(axis=1).fillna(0)
df["no_activity_flag"] = (df[num_cols].sum(axis=1) == 0).astype(int)
df["high_usage_flag"] = (df[num_cols].mean(axis=1) > df[num_cols].mean().mean()).astype(int)

# =========================================================
# 🔥 4. INTERACTION FEATURES (FIXED: Only if meaningful)
# =========================================================

# 🔴 FIX 3: Skip random interaction features (they are meaningless)
# Instead, create meaningful interactions only if needed
# df["interaction_1"] and df["interaction_2"] are REMOVED

# =========================================================
# 💰 5. RFM FEATURES
# =========================================================

# 🔴 FIX 4: RECENCY feature REMOVED (galat logic tha)
# RECENCY remove kar diya kyunki logic sahi nahi tha
# df["recency"] = ... <- REMOVED

# FREQUENCY (total number of non-zero activities) - CORRECT
df["frequency"] = (df[num_cols] > 0).sum(axis=1)

# MONETARY (total value) - CORRECT
df["monetary"] = df[num_cols].sum(axis=1)

# =========================================================
# 📊 6. TIME-BASED FEATURES (BEHAVIOR SHIFT)
# =========================================================

# 🔴 FIX 5: Safe first_activity and last_activity with check
if len(num_cols) > 0:
    df["first_activity"] = df[num_cols[0]]
    df["last_activity"] = df[num_cols[-1]]
    print(f"✅ First column (Month 1?): {num_cols[0]}")
    print(f"✅ Last column (Month 8?): {num_cols[-1]}")
    
    df["activity_trend"] = df["last_activity"] - df["first_activity"]
    # Safe division (already +1 to avoid division by zero)
    df["activity_ratio"] = (df["last_activity"] + 1) / (df["first_activity"] + 1)
else:
    print("⚠️ No numeric columns available for time-based features!")
    df["first_activity"] = 0
    df["last_activity"] = 0
    df["activity_trend"] = 0
    df["activity_ratio"] = 1

# =========================================================
# 📊 7. RATIO FEATURES (VERY IMPORTANT FOR TELECOM)
# =========================================================

df["usage_per_activity"] = df["total_activity"] / (df["frequency"] + 1)
df["revenue_efficiency"] = df["monetary"] / (df["total_activity"] + 1)
df["activity_density"] = df["frequency"] / (len(num_cols) + 1)

# =========================================================
# 🔹 8. FINAL CHECK
# =========================================================

print("\n✅ AFTER ADVANCED FEATURE ENGINEERING")
print("New Shape:", df.shape)

# =========================================================
# 📊 FEATURE LIST REPORT (Updated without bad features)
# =========================================================

all_new_features = [
    "total_activity",
    "avg_usage",
    "usage_variability",
    "no_activity_flag",
    "high_usage_flag",
    "frequency",
    "monetary",
    "activity_trend",
    "activity_ratio",
    "usage_per_activity",
    "revenue_efficiency",
    "activity_density"
]

print("\n📊 GENERATED FEATURES:")
for f in all_new_features:
    if f in df.columns:
        print("✔", f)

# =========================================================
# 📌 FINAL SUMMARY
# =========================================================

print("\n📌 FINAL DATA INFO")
print("Original Shape:", df.shape)

print("\n📊 FINAL SHAPE:", df.shape)

print("\n📉 TOTAL FEATURES ADDED:", len(all_new_features))

print("\n📌 MISSING VALUES AFTER ENGINEERING:", df.isna().sum().sum())

target = "churn"

# ================================
# 1. BASIC CHECK (TARGET INCLUDED?)
# ================================
print("\n📌 Checking if target leaks into features...")

leakage_cols = []

for col in df.columns:
    if col == target:
        continue
    
    # direct leakage check (perfect correlation)
    if df[col].dtype in ['int64', 'float64']:
        corr = df[[col, target]].corr().iloc[0, 1]
        
        if abs(corr) > 0.95:   # VERY HIGH RISK
            leakage_cols.append((col, corr))

# ================================
# 2. STATISTICAL LEAKAGE CHECK
# ================================
print("\n📌 Statistical correlation check (risk detection)...")

risk_features = []

for col in df.select_dtypes(include=['int64','float64']).columns:
    
    if col == target:
        continue
    
    try:
        corr = df[col].corr(df[target])
        
        if abs(corr) > 0.8:
            risk_features.append((col, corr))
            
    except:
        continue

# ================================
# 3. CHECK CONSTANT / ID TYPE FEATURES
# ================================
print("\n📌 Checking ID-like / constant features...")

constant_features = []

for col in df.columns:
    if df[col].nunique() <= 1:
        constant_features.append(col)

# ID-like detection
id_like_features = []

for col in df.columns:
    if "id" in col.lower() or "phone" in col.lower() or "msisdn" in col.lower():
        id_like_features.append(col)

# ================================
# 4. FUTURE / POST-TARGET LEAKAGE CHECK
# ================================
print("\n📌 Checking possible time leakage patterns...")

time_leak_keywords = [
    "after", "future", "next", "post", "target", "churned_date", "exit", "drop_date"
]

time_leak_features = []

for col in df.columns:
    if any(keyword in col.lower() for keyword in time_leak_keywords):
        time_leak_features.append(col)

# ================================
# 5. SUMMARY REPORT
# ================================
print("\n" + "="*70)
print("📊 LEAKAGE ANALYSIS REPORT")
print("="*70)

print("\n🚨 HIGH RISK LEAKAGE (corr > 0.95):")
if leakage_cols:
    for col, corr in leakage_cols:
        print(f"❌ {col} → correlation: {corr:.3f}")
else:
    print("✅ No high-risk leakage found")

print("\n⚠️ MEDIUM RISK FEATURES (corr > 0.8):")
if risk_features:
    for col, corr in risk_features:
        print(f"⚠️ {col} → correlation: {corr:.3f}")
else:
    print("✅ No medium-risk leakage found")

print("\n🧱 CONSTANT FEATURES:")
if constant_features:
    print(constant_features)
else:
    print("✅ No constant features found")

print("\n🆔 ID-LIKE FEATURES:")
if id_like_features:
    print(id_like_features)
else:
    print("✅ No ID-like leakage features found")

print("\n⏳ TIME / FUTURE LEAKAGE FEATURES:")
if time_leak_features:
    print(time_leak_features)
else:
    print("✅ No time-based leakage detected")

# ================================
# 6. FINAL CONCLUSION
# ================================
print("\n" + "="*70)

if (len(leakage_cols) == 0 and 
    len(risk_features) == 0 and 
    len(time_leak_features) == 0):

    print("🎉 FINAL RESULT: NO DATA LEAKAGE DETECTED")
    print("✅ Dataset is SAFE for modeling")

else:
    print("⚠️ WARNING: Potential leakage detected!")
    print("👉 Review highlighted features before training model")

print("\n📊 Dataset Shape:", df.shape)
print("="*70)


# =========================================================
# 🎨 PROFESSIONAL WHITE THEME SETUP
# =========================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

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
print("📊 BOXPLOT: NEW FEATURES VS CHURN ANALYSIS")
print("="*70)

# =========================================================
# 📋 IDENTIFY NEW FEATURES (Updated list)
# =========================================================

# List of newly created features from feature engineering (without bad ones)
new_features_candidates = [
    'total_activity',
    'avg_usage', 
    'usage_variability',
    'no_activity_flag',
    'high_usage_flag',
    'frequency',
    'monetary',
    'activity_trend',
    'activity_ratio',
    'usage_per_activity',
    'revenue_efficiency',
    'activity_density'
]

# Check which features actually exist in dataframe
new_features = [f for f in new_features_candidates if f in df.columns]

print(f"\n✅ New features found: {new_features}")
print(f"📊 Total new features: {len(new_features)}")

# Separate numeric features (for boxplot)
numeric_new_features = [f for f in new_features if df[f].dtype in ['int64', 'float64'] and f not in ['no_activity_flag', 'high_usage_flag']]
categorical_new_features = [f for f in new_features if f in ['no_activity_flag', 'high_usage_flag']]

print(f"📈 Numeric features for boxplot: {numeric_new_features}")
print(f"🏷️ Categorical features (separate analysis): {categorical_new_features}")

# =========================================================
# 📊 CREATE BOXPLOT
# =========================================================

# Calculate number of rows and columns needed
n_features = len(numeric_new_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols if n_features > 0 else 1

if n_features > 0:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, max(5, n_rows * 5)))
    fig.patch.set_facecolor('white')
    
    # Flatten axes for easy indexing
    if n_features > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Store results for summary
    feature_analysis_results = []
    
    for idx, feature in enumerate(numeric_new_features):
        ax = axes_flat[idx]
        
        # Prepare data for boxplot
        data_no_churn = df[df['churn'] == 0][feature].dropna()
        data_churn = df[df['churn'] == 1][feature].dropna()
        
        # Skip if not enough data
        if len(data_no_churn) < 2 or len(data_churn) < 2:
            ax.text(0.5, 0.5, f'Insufficient data for {feature}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate statistics
        median_no_churn = data_no_churn.median()
        median_churn = data_churn.median()
        mean_no_churn = data_no_churn.mean()
        mean_churn = data_churn.mean()
        iqr_no_churn = data_no_churn.quantile(0.75) - data_no_churn.quantile(0.25)
        iqr_churn = data_churn.quantile(0.75) - data_churn.quantile(0.25)
        
        # Mann-Whitney U test for significance
        stat, p_value = mannwhitneyu(data_no_churn, data_churn, alternative='two-sided')
        
        # Create boxplot
        bp = ax.boxplot([data_no_churn, data_churn], 
                         labels=['No Churn', 'Churn'],
                         patch_artist=True, 
                         widths=0.6,
                         medianprops=dict(linewidth=2.5, color='#e94560'),
                         whiskerprops=dict(linewidth=1.5, color='#2c3e50'),
                         capprops=dict(linewidth=1.5, color='#2c3e50'),
                         flierprops=dict(marker='o', markerfacecolor='#ff6b6b', 
                                        markersize=6, alpha=0.5, markeredgecolor='black'))
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('#2E86AB')  # Blue for No Churn
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][0].set_edgecolor('black')
        bp['boxes'][0].set_linewidth(1.5)
        
        bp['boxes'][1].set_facecolor('#A23B72')  # Pink for Churn
        bp['boxes'][1].set_alpha(0.7)
        bp['boxes'][1].set_edgecolor('black')
        bp['boxes'][1].set_linewidth(1.5)
        
        # Add median value labels
        ax.text(1, median_no_churn, f' Median: {median_no_churn:.2f}', 
                va='center', fontsize=9, fontweight='bold', color='#2E86AB')
        ax.text(2, median_churn, f' Median: {median_churn:.2f}', 
                va='center', fontsize=9, fontweight='bold', color='#A23B72')
        
        # Set title with significance indicator
        if p_value < 0.001:
            sig_text = '*** (p < 0.001)'
        elif p_value < 0.01:
            sig_text = '** (p < 0.01)'
        elif p_value < 0.05:
            sig_text = '* (p < 0.05)'
        else:
            sig_text = 'ns (Not Significant)'
        
        ax.set_title(f'{feature}\n{sig_text}', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add statistical info box
        stats_text = f"""
    Statistics:
    ─────────────
    No Churn:
      Mean: {mean_no_churn:.2f}
      Median: {median_no_churn:.2f}
      IQR: {iqr_no_churn:.2f}
    
    Churn:
      Mean: {mean_churn:.2f}
      Median: {median_churn:.2f}
      IQR: {iqr_churn:.2f}
    
    Difference:
      Δ Median: {median_churn - median_no_churn:.2f}
      Δ Mean: {mean_churn - mean_no_churn:.2f}
    """
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                         edgecolor='#2c3e50', alpha=0.9),
                color='#2c3e50')
        
        # Store results
        feature_analysis_results.append({
            'feature': feature,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'median_diff': median_churn - median_no_churn,
            'mean_diff': mean_churn - mean_no_churn,
            'iqr_ratio': iqr_churn / iqr_no_churn if iqr_no_churn != 0 else np.inf
        })
    
    # Remove empty subplots
    for idx in range(len(numeric_new_features), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # =========================================================
    # 📝 ADD SUPPORTIVE DETAILS TEXT
    # =========================================================
    
    supportive_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 BOXPLOT INTERPRETATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔵 BLUE BOX (No Churn)              🔴 PINK BOX (Churn)

📊 WHAT TO LOOK FOR:

1. MEDIAN LINE (Pink Line)
   • Higher median in churn → Feature increases churn risk
   • Lower median in churn → Feature decreases churn risk

2. BOX SIZE (IQR)
   • Larger box = More spread/variability
   • Different sizes → Different behavior patterns

3. STATISTICAL SIGNIFICANCE:
   • *** p < 0.001 → Very Strong evidence
   • ** p < 0.01 → Strong evidence
   • * p < 0.05 → Moderate evidence
   • ns → Not significant

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # Add supportive text at the bottom of figure
    fig.text(0.5, 0.02, supportive_text, transform=fig.transFigure,
             fontsize=8, verticalalignment='bottom', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4f8', 
                      edgecolor='#2c3e50', linewidth=1.5, alpha=0.95),
             color='#2c3e50')
    
    # =========================================================
    # 🎯 MAIN TITLE
    # =========================================================
    
    fig.suptitle('📊 BOXPLOT ANALYSIS: NEW FEATURES vs CHURN STATUS\n'
                 'Comparing Distributions Between Churn and No-Churn Customers',
                 fontsize=14, fontweight='bold', y=0.98, color='#2c3e50')
    
    # Adjust layout to make room for footer text
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.25, 
                        wspace=0.3, hspace=0.4)
    
    # =========================================================
    # 💾 SAVE PLOT
    # =========================================================
    
    save_path = 'reports/figures/boxplot_new_features_vs_churn.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Plot saved: {save_path}")
    
    plt.close()

else:
    print("⚠️ No numeric new features found for boxplot!")
    feature_analysis_results = []

# =========================================================
# 📊 ADDITIONAL: CATEGORICAL FEATURES ANALYSIS (Bar Plot)
# =========================================================

if len(categorical_new_features) > 0:
    print("\n📊 Generating bar plots for categorical features...")
    
    fig2, axes2 = plt.subplots(1, len(categorical_new_features), figsize=(6 * len(categorical_new_features), 5))
    fig2.patch.set_facecolor('white')
    
    if len(categorical_new_features) == 1:
        axes2 = [axes2]
    
    for idx, feature in enumerate(categorical_new_features):
        ax = axes2[idx]
        
        # Calculate churn rate by feature category
        churn_rate = df.groupby(feature)['churn'].mean() * 100
        
        # Create bar plot with safe indexing
        categories = []
        for val in churn_rate.index:
            if val == 0:
                categories.append('No')
            elif val == 1:
                categories.append('Yes')
            else:
                categories.append(str(val))
        
        # Create bars
        bars = ax.bar(categories, churn_rate.values, 
                      color=['#2E86AB', '#A23B72'][:len(categories)], 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, rate in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        # Add count labels (safe way)
        counts = df[feature].value_counts()
        count_text = []
        for val in churn_rate.index:
            if val in counts.index:
                count_text.append(f"{counts[val]}")
            else:
                count_text.append("0")
        
        ax.text(0.5, -0.15, f"Sample Size: {count_text[0]} vs {count_text[1] if len(count_text) > 1 else '0'}", 
               transform=ax.transAxes, ha='center', fontsize=9)
        
        # Styling
        feature_name = feature.replace('_', ' ').title()
        ax.set_title(f'{feature_name}\nChurn Rate Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(churn_rate.values) + 15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Statistical test (Chi-square)
        contingency = pd.crosstab(df[feature], df['churn'])
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        
        # Add significance
        if p_val < 0.05:
            sig_text = f'✓ Significant (p={p_val:.4f})'
            color_status = 'green'
        else:
            sig_text = f'⚠ Not Significant (p={p_val:.4f})'
            color_status = 'red'
        
        ax.text(0.5, 0.95, sig_text, transform=ax.transAxes,
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=color_status))
    
    plt.suptitle('📊 CATEGORICAL FEATURES: CHURN RATE ANALYSIS', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path2 = 'reports/figures/categorical_features_churn_analysis.png'
    plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved: {save_path2}")
    plt.close()

# =========================================================
# 📋 SUMMARY REPORT
# =========================================================

print("\n" + "="*70)
print("📊 FEATURE ANALYSIS SUMMARY")
print("="*70)

if len(feature_analysis_results) > 0:
    print("\n📈 NUMERIC FEATURES (Boxplot Analysis):")
    print("-"*65)
    print(f"{'Feature':<25} {'p-value':<12} {'Significant':<12} {'Median Diff':<12} {'Insight'}")
    print("-"*65)
    
    for result in feature_analysis_results:
        sig_status = "✅ YES" if result['significant'] else "❌ NO"
        insight = ""
        if result['significant']:
            if result['median_diff'] > 0:
                insight = "Higher values → More Churn"
            else:
                insight = "Lower values → More Churn"
        else:
            insight = "Not a good predictor"
        
        print(f"{result['feature']:<25} {result['p_value']:<12.4f} {sig_status:<12} "
              f"{result['median_diff']:<12.2f} {insight}")
    
    print("\n" + "="*70)
    print("🎯 KEY INSIGHTS:")
    print("="*70)
    
    # Find best predictor
    best_predictor = min(feature_analysis_results, key=lambda x: x['p_value'])
    print(f"\n🏆 BEST PREDICTOR: {best_predictor['feature']}")
    print(f"   → p-value: {best_predictor['p_value']:.6f}")
    print(f"   → Median difference: {best_predictor['median_diff']:.2f}")
    if best_predictor['median_diff'] > 0:
        print(f"   → Higher {best_predictor['feature']} = Higher Churn Risk")
    else:
        print(f"   → Lower {best_predictor['feature']} = Higher Churn Risk")
    
    # Count significant features
    significant_count = sum(1 for r in feature_analysis_results if r['significant'])
    print(f"\n📊 Significant Features: {significant_count}/{len(feature_analysis_results)}")
    if len(feature_analysis_results) > 0:
        print(f"   → {significant_count/len(feature_analysis_results)*100:.1f}% of new features are useful for prediction")
else:
    print("\n⚠️ No numeric features were analyzed!")

print("\n" + "="*70)
print("✅ Boxplot analysis complete! Check 'reports/figures/' folder")
print("="*70)

# create directory if not exists
os.makedirs("data/processed", exist_ok=True)

# save final engineered dataset
df.to_csv("data/processed/feature_engineered_dataset.csv", index=False)

print("✅ Feature engineered dataset saved successfully")
print("📁 Path: data/processed/feature_engineered_dataset.csv")
print("📊 Shape:", df.shape)

# =========================================================
# 🎨 NEW PLOT 1A: EFFECT SIZE BAR CHART (Separate)
# =========================================================
print("\n" + "="*70)
print("🎨 PLOT 1A: EFFECT SIZE BAR CHART")
print("="*70)

# Prepare data for feature impact visualization
feature_impact_data = []

for feature in numeric_new_features:
    data_no_churn = df[df['churn'] == 0][feature].dropna()
    data_churn = df[df['churn'] == 1][feature].dropna()
    
    if len(data_no_churn) > 0 and len(data_churn) > 0:
        stat, p_value = mannwhitneyu(data_no_churn, data_churn, alternative='two-sided')
        mean_diff = data_churn.mean() - data_no_churn.mean()
        std_pooled = np.sqrt((data_no_churn.std()**2 + data_churn.std()**2) / 2)
        effect_size = mean_diff / std_pooled if std_pooled > 0 else 0
        
        feature_impact_data.append({
            'feature': feature,
            'p_value': p_value,
            'effect_size': abs(effect_size),
            'median_diff': data_churn.median() - data_no_churn.median(),
            'direction': 'Higher in Churn' if data_churn.median() > data_no_churn.median() else 'Lower in Churn'
        })

impact_df = pd.DataFrame(feature_impact_data)
impact_df = impact_df.sort_values('effect_size', ascending=False)

# Create Effect Size Bar Chart
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_facecolor('#f8f9fa')

top_features = impact_df.head(10)
colors_effect = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_features)))
bars = ax.barh(range(len(top_features)), top_features['effect_size'].values, 
               color=colors_effect, edgecolor='black', linewidth=1.5, height=0.7)

for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['effect_size'] + 0.02, i, f"{row['effect_size']:.3f}", 
            va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values, fontsize=11)
ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
ax.set_title('🎯 FEATURE IMPACT: EFFECT SIZE\n(Larger = More Discriminative between Churn & Non-Churn)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', axis='x')
ax.set_xlim(0, top_features['effect_size'].max() * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation
ax.text(0.98, 0.02, "Effect Size Guide:\n0.2=Small | 0.5=Medium | 0.8=Large", 
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('reports/figures/effect_size_bar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('reports/figures/effect_size_bar_chart.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Effect Size Bar Chart saved!")

# =========================================================
# 🎨 NEW PLOT 1B: STATISTICAL SIGNIFICANCE PLOT (Separate)
# =========================================================
print("\n" + "="*70)
print("🎨 PLOT 1B: STATISTICAL SIGNIFICANCE PLOT")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_facecolor('#f8f9fa')

np.random.seed(42)
x_vals = []
y_vals = []
colors_scatter = []
feature_names = []

for i, (idx, row) in enumerate(impact_df.iterrows()):
    x = 0 if row['p_value'] < 0.05 else 1
    jitter = np.random.normal(0, 0.08)
    x_vals.append(x + jitter)
    y_vals.append(-np.log10(row['p_value'] + 1e-10))
    colors_scatter.append('#2E86AB' if row['p_value'] < 0.05 else '#A23B72')
    feature_names.append(row['feature'])

ax.scatter(x_vals, y_vals, c=colors_scatter, s=120, alpha=0.7, 
           edgecolors='black', linewidth=1.5, zorder=3)

# Add feature labels
for i, name in enumerate(feature_names):
    ax.annotate(name, (x_vals[i], y_vals[i]), 
                xytext=(8, 5), textcoords='offset points', fontsize=8, alpha=0.8)

# Add significance lines
ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, alpha=0.7, label='p = 0.05')
ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', linewidth=2, alpha=0.7, label='p = 0.01')
ax.axhline(y=-np.log10(0.001), color='green', linestyle='--', linewidth=2, alpha=0.7, label='p = 0.001')

ax.set_xticks([0, 1])
ax.set_xticklabels(['✅ Significant\n(p < 0.05)', '❌ Not Significant\n(p ≥ 0.05)'], fontsize=11)
ax.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
ax.set_xlabel('Significance Status', fontsize=12, fontweight='bold')
ax.set_title('📊 STATISTICAL SIGNIFICANCE OF FEATURES\n(Higher = More Significant Difference between Groups)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('reports/figures/statistical_significance_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('reports/figures/statistical_significance_plot.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Statistical Significance Plot saved!")

# =========================================================
# 🎨 NEW PLOT 1C: FEATURE DIRECTION PIE CHART (Separate)
# =========================================================
print("\n" + "="*70)
print("🎨 PLOT 1C: FEATURE DIRECTION PIE CHART")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
ax.set_facecolor('#f8f9fa')

direction_counts = impact_df['direction'].value_counts()
colors_direction = ['#e94560', '#2E86AB']
explode_direction = (0.05, 0.05) if len(direction_counts) > 1 else (0,)

wedges, texts, autotexts = ax.pie(direction_counts.values, labels=direction_counts.index, 
                                   autopct='%1.1f%%', colors=colors_direction[:len(direction_counts)],
                                   explode=explode_direction, startangle=90,
                                   wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.6})

for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')

# Add centre circle for donut
centre_circle = plt.Circle((0, 0), 0.40, fc='white', linewidth=2, edgecolor='#2c3e50')
ax.add_artist(centre_circle)

ax.set_title('📈 FEATURE DIRECTION ANALYSIS\n(Higher in Churn vs Lower in Churn)', 
             fontsize=14, fontweight='bold', pad=20)

# Add interpretation
ax.text(0, -1.2, "🔵 Higher in Churn = Feature increases churn risk\n🔴 Lower in Churn = Feature decreases churn risk",
        ha='center', fontsize=10, fontweight='bold', color='#2c3e50',
        bbox=dict(boxstyle='round', facecolor='#f0f4f8', edgecolor='#2c3e50'))

plt.tight_layout()
plt.savefig('reports/figures/feature_direction_pie.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('reports/figures/feature_direction_pie.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Feature Direction Pie Chart saved!")

# =========================================================
# 🎨 NEW PLOT 1D: TOP 3 FEATURES DISTRIBUTION (Separate)
# =========================================================
print("\n" + "="*70)
print("🎨 PLOT 1D: TOP 3 FEATURES DISTRIBUTION")
print("="*70)

top_3_features = impact_df.head(3)['feature'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
fig.suptitle('📊 TOP 3 FEATURES: CHURN vs NON-CHURN DISTRIBUTION', 
             fontsize=14, fontweight='bold', y=1.02)

for idx, feature in enumerate(top_3_features):
    ax = axes[idx]
    ax.set_facecolor('white')
    
    data_no_churn = df[df['churn'] == 0][feature].dropna()
    data_churn = df[df['churn'] == 1][feature].dropna()
    
    # Histogram with KDE
    sns.histplot(data=data_no_churn, label='No Churn', color='#2E86AB', 
                 alpha=0.5, ax=ax, kde=True, stat='density', bins=30)
    sns.histplot(data=data_churn, label='Churn', color='#e94560', 
                 alpha=0.5, ax=ax, kde=True, stat='density', bins=30)
    
    # Add median lines
    ax.axvline(data_no_churn.median(), color='#2E86AB', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(data_churn.median(), color='#e94560', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add statistics box
    stats_text = f"No Churn Median: {data_no_churn.median():.2f}\nChurn Median: {data_churn.median():.2f}\nDiff: {data_churn.median() - data_no_churn.median():.2f}"
    
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('reports/figures/top3_features_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('reports/figures/top3_features_distribution.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Top 3 Features Distribution saved!")

# =========================================================
# 🎨 NEW PLOT 2: LOLLIPOP CHART - FEATURE CORRELATION WITH TARGET
# =========================================================
print("\n" + "="*70)
print("🎨 NEW PLOT 2: LOLLIPOP CHART - FEATURE CORRELATION WITH TARGET")
print("="*70)

# Calculate correlations with target for all numeric features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'churn' in numeric_features:
    numeric_features.remove('churn')

correlations = []
for feature in numeric_features:
    corr = df[feature].corr(df['churn'])
    correlations.append({
        'feature': feature,
        'correlation': corr,
        'abs_correlation': abs(corr)
    })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('abs_correlation', ascending=False).head(15)

# Separate positive and negative correlations
positive_corr = corr_df[corr_df['correlation'] > 0].sort_values('correlation', ascending=True)
negative_corr = corr_df[corr_df['correlation'] < 0].sort_values('correlation', ascending=False)

fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
ax.set_facecolor('#f8f9fa')

# Plot positive correlations (blue shades)
if len(positive_corr) > 0:
    y_pos = range(len(positive_corr))
    ax.hlines(y=y_pos, xmin=0, xmax=positive_corr['correlation'].values, 
              color='#2E86AB', linewidth=2.5, alpha=0.7)
    ax.scatter(positive_corr['correlation'].values, y_pos, 
               color='#2E86AB', s=120, edgecolor='black', linewidth=1.5, zorder=5, label='Positive Correlation')

# Plot negative correlations (pink shades)
if len(negative_corr) > 0:
    y_neg_start = len(positive_corr)
    y_neg = range(y_neg_start, y_neg_start + len(negative_corr))
    ax.hlines(y=y_neg, xmin=0, xmax=negative_corr['correlation'].values, 
              color='#e94560', linewidth=2.5, alpha=0.7)
    ax.scatter(negative_corr['correlation'].values, y_neg, 
               color='#e94560', s=120, edgecolor='black', linewidth=1.5, zorder=5, label='Negative Correlation')

# Combine labels
all_features = positive_corr['feature'].tolist() + negative_corr['feature'].tolist()
all_y = list(range(len(all_features)))

ax.set_yticks(all_y)
ax.set_yticklabels(all_features, fontsize=10)
ax.set_xlabel('Correlation with Churn', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('📊 LOLLIPOP CHART: Feature Correlation with Churn Target\n'
             'Blue = Positive (Higher value → More Churn) | Pink = Negative (Higher value → Less Churn)',
             fontsize=14, fontweight='bold', pad=20)

# Add vertical line at zero
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Add value labels on points
for i, (idx, row) in enumerate(positive_corr.iterrows()):
    ax.annotate(f"{row['correlation']:.3f}", (row['correlation'], i), 
                xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold', color='#2E86AB')

for i, (idx, row) in enumerate(negative_corr.iterrows()):
    y_pos_neg = len(positive_corr) + i
    ax.annotate(f"{row['correlation']:.3f}", (row['correlation'], y_pos_neg), 
                xytext=(-25, 5), textcoords='offset points', fontsize=8, fontweight='bold', color='#e94560')

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
ax.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='black')

# Add interpretation text
interpretation_text = """
📖 INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 Positive Correlation → Feature increases when customer churns
   Example: Higher value = Higher churn risk

🔴 Negative Correlation → Feature decreases when customer churns
   Example: Higher value = Lower churn risk

|correlation| > 0.1  → Weak but noticeable
|correlation| > 0.3  → Moderate relationship
|correlation| > 0.5  → Strong relationship
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

fig.text(0.02, 0.02, interpretation_text, transform=fig.transFigure,
         fontsize=8, verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f4f8', edgecolor='#2c3e50', linewidth=1.5, alpha=0.95),
         color='#2c3e50')

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('reports/figures/lollipop_feature_correlation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('reports/figures/lollipop_feature_correlation.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Lollipop Chart - Feature Correlation saved!")

# =========================================================
# 📊 FINAL SUMMARY
# =========================================================
print("\n" + "="*70)
print("📊 ALL PLOTS GENERATED SUCCESSFULLY")
print("="*70)
print("\n📁 SAVE LOCATION: reports/figures/")
print("\n📊 NEW PLOTS (Separate Files):")
print("   ┌─────────────────────────────────────────────────────────────────────┐")
print("   │ 1. effect_size_bar_chart.png/pdf                                    │")
print("   │    → Effect Size Bar Chart (Top 10 features)                        │")
print("   │                                                                     │")
print("   │ 2. statistical_significance_plot.png/pdf                           │")
print("   │    → Statistical Significance Scatter Plot                         │")
print("   │                                                                     │")
print("   │ 3. feature_direction_pie.png/pdf                                   │")
print("   │    → Feature Direction Pie Chart                                   │")
print("   │                                                                     │")
print("   │ 4. top3_features_distribution.png/pdf                              │")
print("   │    → Top 3 Features Distribution (Histogram + KDE)                 │")
print("   │                                                                     │")
print("   │ 5. lollipop_feature_correlation.png/pdf                           │")
print("   │    → Lollipop Chart - Correlation with Target                     │")
print("   └─────────────────────────────────────────────────────────────────────┘")

print("\n" + "="*70)
print("🎉 ALL PLOTS COMPLETED SUCCESSFULLY!")
print("="*70)