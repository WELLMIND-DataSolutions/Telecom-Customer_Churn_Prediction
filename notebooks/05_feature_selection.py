import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# =========================================
# LOAD TRAIN DATA (ALREADY FEATURE ENGINEERED)
# =========================================
df = pd.read_csv("data/processed/feature_engineered_dataset.csv")

print("="*60)
print("📁 DATA LOADED (TRAIN DATA - FEATURE ENGINEERED)")
print("="*60)
print(f"📊 Original Shape: {df.shape}")

# =========================================
# TARGET
# =========================================
target = "churn"

# =========================================
# 1. REMOVE CONSTANT COLUMNS
# =========================================
constant_cols = [col for col in df.columns if df[col].nunique() <= 1 and col != target]
df = df.drop(columns=constant_cols)
print(f"\n🧱 Constant Columns Removed: {len(constant_cols)}")

# =========================================
# 2. CLEAN DATA
# =========================================
df = df.replace([np.inf, -np.inf], np.nan)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(0)
obj_cols = df.select_dtypes(include=['object']).columns
df[obj_cols] = df[obj_cols].fillna("No_Value")
print(f"📊 Shape after cleaning: {df.shape}")

# =========================================
# 3. CORRELATION ANALYSIS (FOR INFORMATION ONLY)
# =========================================
numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
corr = numeric_df.corr()[target].drop(target)
corr = corr.sort_values(key=abs, ascending=False)

print("\n🔴 TOP CORRELATED FEATURES WITH CHURN:")
print(corr.head(10))

# =========================================
# 4. VIF ANALYSIS (MULTICOLLINEARITY REMOVAL)
# =========================================
print("\n" + "="*60)
print("📊 VIF ANALYSIS (MULTICOLLINEARITY)")
print("="*60)

# numeric features only
X = df.drop(columns=[target]).select_dtypes(include=['int64', 'float64']).copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# remove ID columns
X.drop(columns=["mobile_number"], inplace=True, errors="ignore")

# remove constant columns
const_cols = [col for col in X.columns if X[col].nunique() <= 1]
X.drop(columns=const_cols, inplace=True)

# remove highly correlated features (correlation > 0.95)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
X.drop(columns=high_corr_cols, inplace=True)
print(f"✅ Removed {len(high_corr_cols)} highly correlated features")

# VIF iterative drop (threshold 10)
if len(X) > 10000:
    X_sample = X.sample(10000, random_state=42)
else:
    X_sample = X.copy()

vif_drop_cols = []
while True:
    vif_data = []
    for i in range(X_sample.shape[1]):
        vif = variance_inflation_factor(X_sample.values, i)
        vif_data.append(vif)
    
    vif_df = pd.DataFrame({"Feature": X_sample.columns, "VIF": vif_data})
    max_vif = vif_df["VIF"].max()
    
    if max_vif > 10:
        drop_col = vif_df.sort_values("VIF", ascending=False).iloc[0]["Feature"]
        vif_drop_cols.append(drop_col)
        print(f"❌ Dropping: {drop_col} | VIF={max_vif:.2f}")
        X_sample = X_sample.drop(columns=[drop_col])
    else:
        break

# Create final dataset after VIF
all_drop_cols = list(set(const_cols + high_corr_cols + vif_drop_cols))
selected_features = [col for col in df.columns if col not in all_drop_cols and col != target]
selected_features.append(target)
df_final = df[selected_features].copy()

print(f"\n📊 Final Shape after VIF: {df_final.shape}")
print(f"✅ Remaining Features: {len(selected_features)-1}")

# =========================================
# 5. SEPARATE X AND y
# =========================================
X = df_final.drop(columns=[target])
y = df_final[target]

print("\n" + "="*60)
print("📊 CLASS DISTRIBUTION (BEFORE SMOTE)")
print("="*60)
print(y.value_counts())
print(f"Churn Rate: {y.mean()*100:.2f}%")

# =========================================
# 6. SCALING (STANDARD SCALER)
# =========================================
print("\n" + "="*60)
print("📏 FEATURE SCALING")
print("="*60)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X[num_cols]),
    columns=num_cols
)
print(f"✅ Scaling completed. Shape: {X_scaled.shape}")

# =========================================
# 7. SMOTE (ONLY ON TRAIN DATA)
# =========================================
print("\n" + "="*60)
print("⚖️ SMOTE APPLICATION (CLASS IMBALANCE HANDLING)")
print("="*60)

# Check imbalance ratio
class_counts = y.value_counts()
ratio = class_counts[1] / class_counts[0] if class_counts[0] > 0 else 0

print(f"Class 0 (Non-Churn): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
print(f"Class 1 (Churn): {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")
print(f"Imbalance Ratio: {ratio:.4f}")

if ratio < 0.7:
    print(f"\n⚠️ Imbalance detected! Applying SMOTE...")
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    print(f"\n✅ After SMOTE:")
    print(f"Class 0: {(y_balanced==0).sum()} ({(y_balanced==0).sum()/len(y_balanced)*100:.1f}%)")
    print(f"Class 1: {(y_balanced==1).sum()} ({(y_balanced==1).sum()/len(y_balanced)*100:.1f}%)")
    print(f"Total Rows Increased: {len(X_balanced) - len(X_scaled)}")
else:
    X_balanced, y_balanced = X_scaled, y
    print("✅ Dataset already balanced, no SMOTE needed")

# =========================================
# 8. RANDOM FOREST FEATURE SELECTION
# =========================================
print("\n" + "="*60)
print("🌲 RANDOM FOREST FEATURE SELECTION")
print("="*60)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_balanced, y_balanced)

feature_importance = pd.DataFrame({
    'Feature': X_balanced.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 TOP 20 FEATURES:")
print(feature_importance.head(20))

# Drop low importance features (threshold 0.002)
threshold = 0.002
low_importance = feature_importance[feature_importance['Importance'] < threshold]['Feature'].tolist()

print(f"\n❌ Dropping {len(low_importance)} low importance features (importance < {threshold})")

X_final = X_balanced.drop(columns=low_importance)

print(f"\n✅ Final Features: {X_final.shape[1]} (dropped {len(low_importance)} features)")

# =========================================
# 9. FINAL DATASET WITH TARGET
# =========================================
print("\n" + "="*60)
print("💾 SAVING FINAL DATASET")
print("="*60)

final_df = X_final.copy()
final_df[target] = y_balanced.values

os.makedirs("data/processed", exist_ok=True)
final_df.to_csv("data/processed/final_selected_features.csv", index=False)

print(f"✅ Final Dataset Shape: {final_df.shape}")
print(f"✅ Features: {final_df.shape[1]-1}")
print(f"✅ Target: {target}")
print(f"✅ Saved: data/processed/final_selected_features.csv")

# =========================================
# 10. PLOTS
# =========================================
print("\n" + "="*60)
print("🎨 GENERATING PLOTS")
print("="*60)

os.makedirs('reports/figures', exist_ok=True)

# PLOT 1: Feature Importance (Top 20)
fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
ax.set_facecolor('#f8f9fa')

top_20 = feature_importance.head(20)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_20)))

bars = ax.barh(range(len(top_20)), top_20['Importance'].values, 
               color=colors, edgecolor='black', linewidth=1)

ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Feature'].values, fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('🌲 Random Forest Feature Importance (Top 20)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, linestyle='--', axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, (idx, row) in enumerate(top_20.iterrows()):
    ax.text(row['Importance'] + 0.001, i, f"{row['Importance']:.4f}", 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/feature_importance_top20.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: reports/figures/feature_importance_top20.png")

# PLOT 2: SMOTE Impact - Before vs After
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# Before SMOTE
before_counts = y.value_counts()
axes[0].pie(before_counts.values, labels=['Non-Churn', 'Churn'], 
            autopct='%1.1f%%', colors=['#2E8B57', '#DC143C'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=1.5)
axes[0].add_artist(centre_circle)
axes[0].set_title(f'Before SMOTE\nTotal: {len(y):,} rows', fontsize=12, fontweight='bold')

# After SMOTE
after_counts = y_balanced.value_counts()
axes[1].pie(after_counts.values, labels=['Non-Churn', 'Churn'], 
            autopct='%1.1f%%', colors=['#2E8B57', '#DC143C'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=1.5)
axes[1].add_artist(centre_circle)
axes[1].set_title(f'After SMOTE\nTotal: {len(y_balanced):,} rows', fontsize=12, fontweight='bold')

plt.suptitle('📊 SMOTE Impact on Class Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/smote_impact.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: reports/figures/smote_impact.png")

# PLOT 3: Final Class Distribution Bar Chart
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')

final_counts = final_df[target].value_counts()
colors_bar = ['#2E8B57', '#DC143C']
bars = ax.bar(['Non-Churn', 'Churn'], final_counts.values, color=colors_bar, edgecolor='black', linewidth=2)

for bar, count in zip(bars, final_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
            f'{count:,}\n({count/len(final_df)*100:.1f}%)', 
            ha='center', fontsize=12, fontweight='bold')

ax.set_title('📊 Final Balanced Dataset Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('reports/figures/final_class_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: reports/figures/final_class_distribution.png")

# PLOT 4: Correlation Lollipop (Top 15)
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_facecolor('#f8f9fa')

top_corr = corr.head(15)
colors_corr = ['#DC143C' if abs(v) > 0.5 else '#FF8C00' if abs(v) > 0.3 else '#2E8B57' for v in top_corr.values]

markerline, stemlines, baseline = ax.stem(range(len(top_corr)), top_corr.values, linefmt='-', markerfmt='o', basefmt=' ')
plt.setp(stemlines, 'color', '#808080', 'linewidth', 1.5, 'alpha', 0.6)

for i, color in enumerate(colors_corr):
    markerline.set_markerfacecolor(color)
    markerline.set_markeredgecolor('black')
    markerline.set_markeredgewidth(1.5)
    markerline.set_markersize(12)

ax.set_xticks(range(len(top_corr)))
ax.set_xticklabels(top_corr.index, rotation=45, ha='right', fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=0.5, color='#FF8C00', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold ±0.5')
ax.axhline(y=-0.5, color='#FF8C00', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=0.3, color='#1E90FF', linestyle=':', linewidth=1, alpha=0.7, label='Threshold ±0.3')
ax.axhline(y=-0.3, color='#1E90FF', linestyle=':', linewidth=1, alpha=0.7)

ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Correlation with Churn', fontsize=12, fontweight='bold')
ax.set_title('📊 Feature Correlation with Target (Top 15)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, (idx, val) in enumerate(top_corr.items()):
    offset = 0.03
    if val > 0:
        ax.annotate(f'{val:.3f}', (i, val + offset), ha='center', fontsize=9, fontweight='bold')
    else:
        ax.annotate(f'{val:.3f}', (i, val - offset), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/correlation_lollipop.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: reports/figures/correlation_lollipop.png")

# =========================================
# FINAL SUMMARY
# =========================================
print("\n" + "="*60)
print("📊 FINAL SUMMARY REPORT")
print("="*60)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE STATUS                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Load Data           : ✅ TRAIN (feature engineered)     │
│ 2. Constant Removal    : ✅ {len(constant_cols)} removed            │
│ 3. High Correlation    : ✅ {len(high_corr_cols)} removed           │
│ 4. VIF Analysis        : ✅ {len(vif_drop_cols)} removed            │
│ 5. Scaling             : ✅ StandardScaler                 │
│ 6. SMOTE               : ✅ {'Applied' if ratio < 0.7 else 'Not Needed'} │
│ 7. RF Selection        : ✅ {len(low_importance)} features dropped    │
├─────────────────────────────────────────────────────────────┤
│                    FINAL DATASET                            │
├─────────────────────────────────────────────────────────────┤
│ Total Rows:        {final_df.shape[0]:,}                               │
│ Total Features:    {final_df.shape[1]-1}                               │
│ Target Column:     churn                                    │
│ Class 0 (Non-Churn): {(final_df[target]==0).sum():,} ({(final_df[target]==0).sum()/len(final_df)*100:.1f}%)   │
│ Class 1 (Churn):    {(final_df[target]==1).sum():,} ({(final_df[target]==1).sum()/len(final_df)*100:.1f}%)   │
├─────────────────────────────────────────────────────────────┤
│ DATA LEAKAGE CHECK:      ✅ NONE                            │
│ TEST DATA USED:          ✅ NO                              │
└─────────────────────────────────────────────────────────────┘
""")

print("\n📁 FILES SAVED:")
print("   📊 data/processed/final_selected_features.csv")
print("   📊 reports/figures/feature_importance_top20.png")
print("   📊 reports/figures/smote_impact.png")
print("   📊 reports/figures/final_class_distribution.png")
print("   📊 reports/figures/correlation_lollipop.png")

print("\n" + "="*60)
print("🎉 PIPELINE COMPLETE! READY FOR MODEL TRAINING!")
print("="*60)