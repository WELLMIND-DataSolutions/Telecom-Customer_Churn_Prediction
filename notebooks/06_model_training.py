# =========================================
# CHURN PREDICTION - MODEL COMPARISON (No Split, Only CV)
# =========================================

import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 1. LOAD DATA
# =========================================
df = pd.read_csv("data/processed/final_selected_features.csv")

# Separate features and target
X = df.drop(columns=['churn'])
y = df['churn']

print("✅ Data loaded successfully")
print(f"Shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Churn rate: {y.mean():.2%}")

# =========================================
# 2. DEFINE MODELS WITH HYPERPARAMETER GRIDS
# =========================================
models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, 
                               scale_pos_weight=len(y[y==0])/len(y[y==1])),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.6, 0.8]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.05],
            'num_leaves': [31, 50]
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
        'params': {
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.01, 0.05]
        }
    }
}

# =========================================
# 3. CROSS-VALIDATION SETUP
# =========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics to track
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# =========================================
# 4. TRAIN AND EVALUATE EACH MODEL (Using CV only)
# =========================================
results = []
best_f1 = 0
best_model_name = None
best_model_obj = None
training_times = {}

print("\n" + "="*70)
print("STARTING MODEL EVALUATION (5-Fold Cross-Validation)")
print("="*70)

for name, mp in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)
    
    start_time = time.time()
    
    # Hyperparameter tuning with RandomizedSearchCV (3-fold CV)
    random_search = RandomizedSearchCV(
        mp['model'], mp['params'], 
        n_iter=8, cv=StratifiedKFold(3), 
        scoring='f1', random_state=42, n_jobs=-1
    )
    random_search.fit(X, y)  # Using FULL data for tuning
    
    training_time = time.time() - start_time
    training_times[name] = training_time
    
    best_model = random_search.best_estimator_
    print(f"Best params: {random_search.best_params_}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Cross-validation scores on FULL data
    cv_results = cross_validate(best_model, X, y, cv=cv, scoring=scoring)
    
    print(f"\n📊 5-Fold Cross-Validation Results:")
    print(f"   Accuracy : {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
    print(f"   Precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
    print(f"   Recall   : {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
    print(f"   F1 Score : {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
    
    # Store results
    results.append({
        'Model': name,
        'CV Accuracy': cv_results['test_accuracy'].mean(),
        'CV Accuracy Std': cv_results['test_accuracy'].std(),
        'CV Precision': cv_results['test_precision'].mean(),
        'CV Recall': cv_results['test_recall'].mean(),
        'CV F1 Score': cv_results['test_f1'].mean(),
        'CV F1 Std': cv_results['test_f1'].std(),
        'Training Time (s)': training_time
    })
    
    # Track best model based on F1 Score
    current_f1 = cv_results['test_f1'].mean()
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_model_name = name
        best_model_obj = best_model

# =========================================
# 5. RESULTS COMPARISON TABLE
# =========================================
print("\n" + "="*80)
print("FINAL COMPARISON OF ALL MODELS (5-Fold CV)")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='CV F1 Score', ascending=False)
print(results_df.round(4).to_string(index=False))

# Save results to CSV
results_df.to_csv('data/processed/model_comparison_results_cv.csv', index=False)
print("\n✅ Results saved to: data/processed/model_comparison_results_cv.csv")

# =========================================
# 6. VISUALIZATIONS
# =========================================

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Figure 1: F1 Score Comparison with Error Bars
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(results_df['Model'], results_df['CV F1 Score'], yerr=results_df['CV F1 Std'], 
               capsize=5, color='skyblue', edgecolor='black', error_kw={'elinewidth': 2, 'markeredgewidth': 2})
ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_title('Model Comparison - 5-Fold CV F1 Score (± Std Dev)', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1])
for bar, score in zip(bars, results_df['CV F1 Score']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{score:.3f}', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/figures/cv_f1_score_comparison.png', dpi=150)
print("✅ CV F1 Score plot saved to: data/processed/cv_f1_score_comparison.png")

# Figure 2: Precision, Recall, F1 (Grouped Bar)
fig2, ax2 = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df['Model']))
width = 0.25

ax2.bar(x - width, results_df['CV Precision'], width, label='Precision', color='lightcoral')
ax2.bar(x, results_df['CV Recall'], width, label='Recall', color='lightgreen')
ax2.bar(x + width, results_df['CV F1 Score'], width, label='F1 Score', color='skyblue')

ax2.set_xlabel('Models', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Model Comparison - Precision, Recall, F1 Score (5-Fold CV)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax2.legend()
ax2.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('reports/figures/cv_precision_recall_f1_comparison.png', dpi=150)
print("✅ CV Precision-Recall-F1 plot saved to: data/processed/cv_precision_recall_f1_comparison.png")

# Figure 3: Training Time Comparison
fig3, ax3 = plt.subplots(figsize=(10, 6))
bars = ax3.bar(results_df['Model'], results_df['Training Time (s)'], color='lightblue', edgecolor='black')
ax3.set_xlabel('Models', fontsize=12)
ax3.set_ylabel('Training Time (seconds)', fontsize=12)
ax3.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
for bar, time_val in zip(bars, results_df['Training Time (s)']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/figures/cv_training_time_comparison.png', dpi=150)
print("✅ Training time plot saved to: data/processed/cv_training_time_comparison.png")

# Figure 4: Accuracy with Error Bars
fig4, ax4 = plt.subplots(figsize=(10, 6))
bars = ax4.bar(results_df['Model'], results_df['CV Accuracy'], yerr=results_df['CV Accuracy Std'], 
               capsize=5, color='lightgreen', edgecolor='black', error_kw={'elinewidth': 2})
ax4.set_xlabel('Models', fontsize=12)
ax4.set_ylabel('Accuracy', fontsize=12)
ax4.set_title('Model Comparison - 5-Fold CV Accuracy (± Std Dev)', fontsize=14, fontweight='bold')
ax4.set_ylim([0, 1])
for bar, acc in zip(bars, results_df['CV Accuracy']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/figures/cv_accuracy_comparison.png', dpi=150)
print("✅ CV Accuracy plot saved to: data/processed/cv_accuracy_comparison.png")

# =========================================
# 5. RECOMMEND BEST MODEL
# =========================================
print("\n" + "="*70)
print(f"🏆 RECOMMENDED BEST MODEL: {best_model_name}")
print("="*70)
print(f"CV F1 Score  : {best_f1:.4f}")
print(f"CV Precision : {results_df[results_df['Model']==best_model_name]['CV Precision'].values[0]:.4f}")
print(f"CV Recall    : {results_df[results_df['Model']==best_model_name]['CV Recall'].values[0]:.4f}")
print(f"CV Accuracy  : {results_df[results_df['Model']==best_model_name]['CV Accuracy'].values[0]:.4f}")

print("\n" + "="*70)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📁 Check 'data/processed/' folder for:")
print("   ✅ model_comparison_results_cv.csv")
print("   ✅ cv_f1_score_comparison.png")
print("   ✅ cv_precision_recall_f1_comparison.png")
print("   ✅ cv_training_time_comparison.png")
print("   ✅ cv_accuracy_comparison.png")