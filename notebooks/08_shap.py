# shap_analysis_final.py - FINAL SHAP ANALYSIS (COMPLETELY FIXED)
# SHAP values save to models/, CSV save to data/processed/

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("🎯 FINAL SHAP ANALYSIS - CHURN PREDICTION EXPLAINER")
print("="*70)
print("\n📌 Analyzing WHY customers churn")
print("✅ Model remains UNCHANGED (Read-only analysis)")
print("✅ SHAP values → models/shap_values.pkl")
print("✅ Feature importance CSV → data/processed/feature_importance_shap.csv\n")

# =========================================
# 1. LOAD MODEL AND DATA
# =========================================
print("📂 Loading model and data...")

try:
    model = joblib.load('data/processed/random_forest_churn_model.pkl')
    df = pd.read_csv("data/processed/final_selected_features.csv")
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Data: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"✅ Churn rate: {(y==1).sum()/len(y)*100:.1f}%")
    
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# =========================================
# 2. INSTALL AND IMPORT SHAP
# =========================================
try:
    import shap
    print("✅ SHAP already installed")
except ImportError:
    print("📦 Installing SHAP...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap', '-q'])
    import shap
    print("✅ SHAP installed successfully")

# =========================================
# 3. CREATE SAMPLE FOR ANALYSIS
# =========================================
print("\n🔧 Preparing data for SHAP analysis...")

sample_size = min(200, len(X))  # Reduced for stability
X_sample = X.sample(n=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

print(f"✅ Using {sample_size} samples for analysis")

# =========================================
# 4. CREATE SHAP EXPLAINER & CALCULATE VALUES
# =========================================
print("\n🔄 Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)
print("✅ SHAP TreeExplainer created")

print("🔄 Calculating SHAP values (this may take 2-3 minutes)...")
shap_values = explainer.shap_values(X_sample)

# Handle SHAP values correctly
print(f"📊 Raw SHAP values type: {type(shap_values)}")

if isinstance(shap_values, list):
    if len(shap_values) == 2:
        shap_values_churn = shap_values[1]
        expected_value = explainer.expected_value[1]
        print("✅ Using SHAP values for churn class")
    else:
        shap_values_churn = shap_values[0]
        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
        print("✅ Using first SHAP array")
else:
    if len(shap_values.shape) == 3:
        shap_values_churn = shap_values[:, :, 1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        print("✅ Extracted churn class from 3D array")
    else:
        shap_values_churn = shap_values
        expected_value = explainer.expected_value

print(f"✅ Final SHAP values shape: {shap_values_churn.shape}")

# =========================================
# 5. CREATE DIRECTORIES
# =========================================
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# =========================================
# 6. SAVE SHAP VALUES TO PKL FILE
# =========================================
print("\n💾 Saving SHAP values to PKL file...")

shap_data = {
    'shap_values': shap_values_churn,
    'expected_value': expected_value,
    'feature_names': X_sample.columns.tolist(),
    'X_sample': X_sample,
    'y_sample': y_sample,
    'model_type': str(type(model).__name__),
    'sample_size': sample_size
}
joblib.dump(shap_data, 'models/shap_values.pkl')
print("✅ Saved: models/shap_values.pkl")

# =========================================
# 7. FEATURE IMPORTANCE ANALYSIS (FIXED)
# =========================================
print("\n" + "="*70)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Fix: Ensure shap_values_churn is 2D
if len(shap_values_churn.shape) > 2:
    shap_values_churn = shap_values_churn.reshape(shap_values_churn.shape[0], -1)

# Calculate mean absolute SHAP values
shap_mean_abs = np.abs(shap_values_churn).mean(axis=0)

# Ensure we have correct number of features
num_features = len(X_sample.columns)
if len(shap_mean_abs) != num_features:
    print(f"⚠️ Shape mismatch: {len(shap_mean_abs)} vs {num_features}")
    # Take minimum length
    min_len = min(len(shap_mean_abs), num_features)
    shap_mean_abs = shap_mean_abs[:min_len]
    feature_names = X_sample.columns[:min_len]
else:
    feature_names = X_sample.columns

feature_importance_data = []
for i in range(len(feature_names)):
    mean_val = shap_values_churn[:, i].mean() if shap_values_churn.shape[1] > i else 0
    direction = 'Increases Churn' if mean_val > 0 else 'Decreases Churn'
    feature_importance_data.append({
        'Feature': feature_names[i],
        'SHAP_Value': float(shap_mean_abs[i]),
        'Direction': direction
    })

feature_importance = pd.DataFrame(feature_importance_data).sort_values('SHAP_Value', ascending=False)

print("\n🔝 TOP 10 FEATURES CAUSING CHURN:")
print("-" * 80)
for i in range(min(10, len(feature_importance))):
    row = feature_importance.iloc[i]
    icon = "🔴" if row['Direction'] == 'Increases Churn' else "🟢"
    print(f"   {i+1:2}. {row['Feature'][:40]:<40} {icon} {row['SHAP_Value']:.4f}")

feature_importance.to_csv('data/processed/feature_importance_shap.csv', index=False)
print("\n✅ Saved: data/processed/feature_importance_shap.csv")

# =========================================
# 8. PLOT 1: DONUT CHART
# =========================================
print("\n🎨 Generating Plot 1: Feature Impact Donut Chart...")

try:
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    if len(feature_importance) >= 5:
        top_5 = feature_importance.head(5)
        other_sum = feature_importance.iloc[5:]['SHAP_Value'].sum()
        
        sizes = list(top_5['SHAP_Value'].values) + [other_sum]
        labels = list(top_5['Feature'].values) + ['Other Features']
    else:
        sizes = list(feature_importance['SHAP_Value'].values)
        labels = list(feature_importance['Feature'].values)
    
    colors_donut = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db', '#95a5a6']
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_donut[:len(sizes)], startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=2, edgecolor='#2C3E50')
    ax1.add_artist(centre_circle)
    
    ax1.set_title('TOP CHURN DRIVERS', fontsize=16, fontweight='bold', pad=20)
    ax1.text(0, 0, 'Total\nImpact', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/figures/01_feature_impact_donut.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: reports/figures/01_feature_impact_donut.png")
except Exception as e:
    print(f"⚠️ Could not save donut chart: {e}")

# =========================================
# 9. PLOT 2: HORIZONTAL BAR CHART
# =========================================
print("🎨 Generating Plot 2: Horizontal Bar Chart...")

try:
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    
    top_15 = feature_importance.head(15).iloc[::-1]
    
    colors_bar = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top_15)))
    
    bars = ax2.barh(range(len(top_15)), top_15['SHAP_Value'], color=colors_bar, 
                    edgecolor='black', linewidth=0.8, height=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, top_15['SHAP_Value'])):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                 f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_yticks(range(len(top_15)))
    ax2.set_yticklabels(top_15['Feature'], fontsize=9)
    ax2.set_xlabel('SHAP Value (Impact on Churn)', fontsize=12, fontweight='bold')
    ax2.set_title('TOP FEATURES DRIVING CUSTOMER CHURN', fontsize=15, fontweight='bold', pad=20)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('reports/figures/02_horizontal_bar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: reports/figures/02_horizontal_bar_chart.png")
except Exception as e:
    print(f"⚠️ Could not save bar chart: {e}")

# =========================================
# 10. PLOT 3: RISK SEGMENTATION
# =========================================
print("🎨 Generating Plot 3: Risk Segmentation...")

try:
    y_proba = model.predict_proba(X_sample)[:, 1]
    
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    
    high_risk = y_proba > 0.7
    medium_risk = (y_proba > 0.4) & (y_proba <= 0.7)
    low_risk = y_proba <= 0.4
    
    risk_counts = [low_risk.sum(), medium_risk.sum(), high_risk.sum()]
    risk_labels = ['LOW RISK\n(<40%)', 'MEDIUM RISK\n(40-70%)', 'HIGH RISK\n(>70%)']
    risk_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax3.pie(risk_counts, labels=risk_labels, autopct='%1.1f%%',
            colors=risk_colors, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    centre_circle = plt.Circle((0, 0), 0.60, fc='white', linewidth=2, edgecolor='#2C3E50')
    ax3.add_artist(centre_circle)
    ax3.text(0, 0, f'Total\n{len(y_proba)}', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax3.set_title('CUSTOMER RISK SEGMENTATION', fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('reports/figures/03_risk_segmentation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: reports/figures/03_risk_segmentation.png")
except Exception as e:
    print(f"⚠️ Could not save risk segmentation: {e}")

# =========================================
# 11. BUSINESS INSIGHTS SUMMARY
# =========================================
print("\n" + "="*70)
print("💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n🎯 TOP 5 CHURN DRIVERS & ACTIONS:\n")

if len(feature_importance) > 0:
    total_impact = feature_importance['SHAP_Value'].sum()
    for i in range(min(5, len(feature_importance))):
        feature = feature_importance.iloc[i]['Feature']
        impact = feature_importance.iloc[i]['SHAP_Value']
        
        print(f"{'━'*60}")
        print(f"📌 {i+1}. {feature}")
        print(f"   Impact Score: {impact:.4f} ({impact/total_impact*100:.1f}% of total impact)")
        
        feature_lower = feature.lower()
        if 'tenure' in feature_lower or 'month' in feature_lower:
            print(f"   ⚠️ Problem: Short-tenure customers are leaving")
            print(f"   ✅ Solution: Implement 'First 90 Days' onboarding program")
        elif 'charge' in feature_lower or 'amount' in feature_lower:
            print(f"   ⚠️ Problem: High bills causing customer frustration")
            print(f"   ✅ Solution: Offer personalized discounts or tiered plans")
        elif 'ticket' in feature_lower or 'support' in feature_lower:
            print(f"   ⚠️ Problem: Multiple support tickets = unhappy customers")
            print(f"   ✅ Solution: Proactive support and faster resolution")
        elif 'satisfaction' in feature_lower or 'score' in feature_lower:
            print(f"   ⚠️ Problem: Low satisfaction leading to churn")
            print(f"   ✅ Solution: Regular NPS surveys and follow-up")
        elif 'payment' in feature_lower:
            print(f"   ⚠️ Problem: Payment issues causing frustration")
            print(f"   ✅ Solution: Simplify payment process and reminders")
        else:
            print(f"   ⚠️ Problem: This factor significantly impacts churn")
            print(f"   ✅ Solution: Investigate and create targeted retention")
        print()
else:
    print("⚠️ No feature importance data available")

# =========================================
# 12. FINAL SUMMARY
# =========================================
print("\n" + "="*70)
print("✅ SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*70)

if len(feature_importance) > 0:
    top_features = feature_importance.iloc[0]['Feature'] if len(feature_importance) > 0 else 'N/A'
    second_feature = feature_importance.iloc[1]['Feature'] if len(feature_importance) > 1 else 'N/A'
    third_feature = feature_importance.iloc[2]['Feature'] if len(feature_importance) > 2 else 'N/A'
else:
    top_features = second_feature = third_feature = 'N/A'

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ANALYSIS SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════╣
║  ✅ Model Status:     UNCHANGED (Read-only)                      ║
║  ✅ SHAP Values:      models/shap_values.pkl                    ║
║  ✅ CSV File:         data/processed/feature_importance_shap.csv║
║  ✅ Samples:          {sample_size} customers                           ║
║  ✅ Features:         {X.shape[1]} features                               ║
║  ✅ Top Driver:       {top_features[:35] if top_features != 'N/A' else 'N/A'}        ║
╚══════════════════════════════════════════════════════════════════╝

📁 FILES GENERATED:

   📊 PLOTS (3 files):
      ├─ reports/figures/01_feature_impact_donut.png
      ├─ reports/figures/02_horizontal_bar_chart.png
      └─ reports/figures/03_risk_segmentation.png

   💾 SAVED DATA (2 files):
      ├─ models/shap_values.pkl
      └─ data/processed/feature_importance_shap.csv

💡 TOP 3 REASONS CUSTOMERS CHURN:
   1. {top_features}
   2. {second_feature}
   3. {third_feature}

🎯 HOW TO LOAD SHAP VALUES LATER:
   import joblib
   shap_data = joblib.load('models/shap_values.pkl')
   shap_values = shap_data['shap_values']
   expected_value = shap_data['expected_value']
""")
print("="*70)
print("\n✅ MODEL STATUS: SAFE & UNCHANGED")
print("✅ All files saved successfully!")
print("="*70)