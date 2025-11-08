import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb


warnings.filterwarnings('ignore')

data= pd.read_csv("C:/Users/Kareem/Desktop/cumulative_2025.10.02_20.38.17.csv")

drop_cols = [
    "koi_longp", "koi_ingress", "koi_model_dof", "koi_model_chisq", "koi_sage",

    "rowid", "kepoi_name", "kepid", "kepler_name",

    "koi_pdisposition", "koi_score", "koi_time0bk",

    "koi_comment", "koi_limbdark_mod", "koi_parm_prov",
    "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs",
    "koi_tce_delivname", "koi_sparprov", "koi_vet_stat",
    "koi_vet_date", "koi_disp_prov",

    "koi_ldm_coeff3", "koi_ldm_coeff4"
]

df_cleaned = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")


for i in df_cleaned.columns.tolist():
  if df_cleaned[i].nunique() == 1 or df_cleaned[i].nunique() == 0:
    df_cleaned.drop(i,axis=1,inplace=True)
for i in df_cleaned.columns.tolist():
  print("No. of unique values in",i,"is",df_cleaned[i].nunique())
numerical_features = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_features:
    if df_cleaned[col].isnull().sum() > 0: 
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())  # Fill with median

# Feature Engineering
def create_advanced_features(df):
    df_new = df.copy()
    
    # === Orbital Mechanics Features ===
    if 'koi_period' in df.columns and 'koi_teq' in df.columns:
        df_new['temp_period_ratio'] = df_new['koi_teq'] / (df_new['koi_period'] + 1e-10)
        df_new['temp_period_product'] = df_new['koi_teq'] * df_new['koi_period']
    
    if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
        df_new['planet_star_radius_ratio'] = df_new['koi_prad'] / (df_new['koi_srad'] + 1e-10)
        df_new['radius_difference'] = np.abs(df_new['koi_prad'] - df_new['koi_srad'])
    
    if 'koi_period' in df.columns:
        df_new['log_period'] = np.log1p(df_new['koi_period'])
        df_new['sqrt_period'] = np.sqrt(df_new['koi_period'])
        df_new['period_squared'] = df_new['koi_period'] ** 2
        df_new['inv_period'] = 1 / (df_new['koi_period'] + 1e-10)
    
    if 'koi_depth' in df.columns:
        df_new['log_depth'] = np.log1p(df_new['koi_depth'])
        df_new['sqrt_depth'] = np.sqrt(df_new['koi_depth'])
    
    if 'koi_duration' in df.columns and 'koi_period' in df.columns:
        df_new['duration_period_ratio'] = df_new['koi_duration'] / (df_new['koi_period'] + 1e-10)
        df_new['duration_period_product'] = df_new['koi_duration'] * df_new['koi_period']
    
    # === Stellar Properties ===
    if 'koi_steff' in df.columns:
        df_new['log_steff'] = np.log1p(df_new['koi_steff'])
        df_new['steff_squared'] = df_new['koi_steff'] ** 2
    
    if 'koi_slogg' in df.columns and 'koi_srad' in df.columns:
        df_new['logg_radius_interaction'] = df_new['koi_slogg'] * df_new['koi_srad']
        df_new['stellar_density_proxy'] = df_new['koi_slogg'] / (df_new['koi_srad'] ** 2 + 1e-10)
    
    if 'koi_slogg' in df.columns:
        df_new['log_slogg'] = np.log1p(df_new['koi_slogg'] + 10)
    
    # === Transit Properties ===
    if 'koi_impact' in df.columns:
        df_new['impact_squared'] = df_new['koi_impact'] ** 2
        df_new['impact_sqrt'] = np.sqrt(np.abs(df_new['koi_impact']))
        df_new['impact_cubed'] = df_new['koi_impact'] ** 3
    
    # === Insolation & Habitability ===
    if 'koi_insol' in df.columns:
        df_new['log_insol'] = np.log1p(df_new['koi_insol'])
        df_new['sqrt_insol'] = np.sqrt(df_new['koi_insol'])
        df_new['habitable_zone'] = ((df_new['koi_insol'] >= 0.25) & 
                                     (df_new['koi_insol'] <= 4.0)).astype(int)
        df_new['conservative_hz'] = ((df_new['koi_insol'] >= 0.5) & 
                                      (df_new['koi_insol'] <= 2.0)).astype(int)
    
  # === Signal Quality ===
    if 'koi_model_snr' in df.columns:
        df_new['log_snr'] = np.log1p(df_new['koi_model_snr'])
        df_new['high_snr'] = (df_new['koi_model_snr'] > 50).astype(int)
        df_new['very_high_snr'] = (df_new['koi_model_snr'] > 100).astype(int)
        df_new['snr_squared'] = df_new['koi_model_snr'] ** 2
    
    # === Earth Similarity Index ===
    if 'koi_prad' in df.columns and 'koi_teq' in df.columns:
        df_new['earth_similarity'] = 1 / (1 + np.abs(df_new['koi_prad'] - 1) + 
                                          np.abs(df_new['koi_teq'] - 288)/100)
    
    if 'koi_prad' in df.columns and 'koi_insol' in df.columns:
        df_new['earth_like_proxy'] = 1 / (1 + np.abs(df_new['koi_prad'] - 1) + 
                                          np.abs(df_new['koi_insol'] - 1))
    
    # === Combined Features ===
    if 'koi_depth' in df.columns and 'koi_duration' in df.columns:
        df_new['depth_duration_ratio'] = df_new['koi_depth'] / (df_new['koi_duration'] + 1e-10)
    
    if 'koi_prad' in df.columns and 'koi_period' in df.columns:
        df_new['radius_period_ratio'] = df_new['koi_prad'] / (df_new['koi_period'] + 1e-10)
    
    return df_new




# ============================================
# 2. Data Preparation
# ============================================

print(" COMPREHENSIVE EXOPLANET CLASSIFICATION PIPELINE")
print("   Including: XGBoost, LightGBM, CatBoost, Neural Networks")


print("\n[STEP 1] Data Preparation...")
le = LabelEncoder()
df_cleaned['koi_disposition_encoded'] = le.fit_transform(df_cleaned['koi_disposition'])

print(f"Target classes: {le.classes_}")
print(f"Class distribution:\n{df_cleaned['koi_disposition'].value_counts()}")

# Create engineered features
print("\n[STEP 2] Feature Engineering...")
df_engineered = create_advanced_features(df_cleaned)

# Select numerical features
numerical_features = df_engineered.select_dtypes(include=['float64', 'int64']).columns
numerical_features = [col for col in numerical_features if col != 'koi_disposition_encoded']

original_count = len([c for c in df_cleaned.columns if c in numerical_features])
new_count = len(numerical_features)

print(f" Original features: {original_count}")
print(f" Engineered features: {new_count}")
print(f" New features added: {new_count - original_count}")

# Handle missing values and infinities
X = df_engineered[numerical_features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = df_engineered['koi_disposition_encoded']

# ============================================
# 3. Train-Test Split & Scaling
# ============================================
print("\n[STEP 3] Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Training samples: {X_train.shape[0]}")
print(f" Testing samples: {X_test.shape[0]}")

# ============================================
# 4. Feature Selection
# ============================================
print("\n[STEP 4] Feature Selection...")

selector_f = SelectKBest(f_classif, k='all')
selector_f.fit(X_train_scaled, y_train)

selector_mi = SelectKBest(mutual_info_classif, k='all')
selector_mi.fit(X_train_scaled, y_train)

feature_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': selector_f.scores_,
    'mi_score': selector_mi.scores_
})
feature_scores['combined_score'] = (feature_scores['f_score'] / feature_scores['f_score'].max() + 
                                     feature_scores['mi_score'] / feature_scores['mi_score'].max())
feature_scores = feature_scores.sort_values('combined_score', ascending=False)



k_best = min(100, len(X.columns))
top_features = feature_scores.head(k_best)['feature'].values
X_train_selected = X_train_scaled[:, [X.columns.get_loc(f) for f in top_features]]
X_test_selected = X_test_scaled[:, [X.columns.get_loc(f) for f in top_features]]

# ============================================
# 5. Polynomial Features
# ============================================
print("\n[STEP 5] Creating Polynomial Interactions...")

top_10_features = feature_scores.head(8)['feature'].values
print(top_10_features)
top_10_idx = [X.columns.get_loc(f) for f in top_10_features]

X_train_top = X_train_scaled[:, top_10_idx]
X_test_top = X_test_scaled[:, top_10_idx]

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_top)
X_test_poly = poly.transform(X_test_top)

X_train_combined = np.hstack([X_train_selected, X_train_poly])
X_test_combined = np.hstack([X_test_selected, X_test_poly])

print(f" Combined features: {X_train_combined.shape[1]}")

# ============================================
# 6. Model Training & Hyperparameter Tuning
# ============================================
print(" MODEL TRAINING & OPTIMIZATION")

results = []
predictions = {}
trained_models = {}

# === 1. Random Forest ===
print("\n[1/10] Training Random Forest...")
rf_params = {
    'n_estimators': [500, 1000],
    'max_depth': [10, 12, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1
                                 ,max_depth=8,n_estimators=1000)
rf_random = RandomizedSearchCV(rf_base, rf_params, n_iter=10, cv=5, 
                               scoring='accuracy', random_state=42, n_jobs=-1, verbose=0)
rf_random.fit(X_train_combined, y_train)
best_rf = rf_random.best_estimator_
y_pred_rf = best_rf.predict(X_test_combined)
acc_rf = accuracy_score(y_test, y_pred_rf)
results.append({'Model': 'Random Forest', 'Accuracy': acc_rf})
predictions['Random Forest'] = y_pred_rf
trained_models['Random Forest'] = best_rf
print(f"✓ Random Forest Accuracy: {acc_rf:.4f}")



# === 3. XGBoost ===

print("\n[3/10] Training XGBoost...")
xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=1,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
xgb_model.fit(X_train_combined, y_train)
y_pred_xgb = xgb_model.predict(X_test_combined)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
results.append({'Model': 'XGBoost', 'Accuracy': acc_xgb})
predictions['XGBoost'] = y_pred_xgb
trained_models['XGBoost'] = xgb_model
print(f"✓ XGBoost Accuracy: {acc_xgb:.4f}")

# === 4. LightGBM ===
print("\n[4/10] Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
lgb_model.fit(X_train_combined, y_train)
y_pred_lgb = lgb_model.predict(X_test_combined)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
results.append({'Model': 'LightGBM', 'Accuracy': acc_lgb})
predictions['LightGBM'] = y_pred_lgb
trained_models['LightGBM'] = lgb_model
print(f" LightGBM Accuracy: {acc_lgb:.4f}")

# === 5. CatBoost ===
print("\n[5/10] Training CatBoost...")
cat_model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=0
    )
cat_model.fit(X_train_combined, y_train)
y_pred_cat = cat_model.predict(X_test_combined)
acc_cat = accuracy_score(y_test, y_pred_cat)
results.append({'Model': 'CatBoost', 'Accuracy': acc_cat})
predictions['CatBoost'] = y_pred_cat
trained_models['CatBoost'] = cat_model
print(f" CatBoost Accuracy: {acc_cat:.4f}")

# === 6. AdaBoost ===
print("\n[6/10] Training AdaBoost...")
ada_model = AdaBoostClassifier(
    n_estimators=500,
    learning_rate=0.1,
    random_state=42
)
ada_model.fit(X_train_combined, y_train)
y_pred_ada = ada_model.predict(X_test_combined)
acc_ada = accuracy_score(y_test, y_pred_ada)
results.append({'Model': 'AdaBoost', 'Accuracy': acc_ada})
predictions['AdaBoost'] = y_pred_ada
trained_models['AdaBoost'] = ada_model
print(f"✓ AdaBoost Accuracy: {acc_ada:.4f}")



# === 10. Baseline ===
print("\n[10/10] Training Baseline (for comparison)...")
baseline_rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42, n_jobs=-1)
baseline_rf.fit(X_train_selected, y_train)
y_pred_baseline = baseline_rf.predict(X_test_selected)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
results.append({'Model': 'Baseline RF', 'Accuracy': acc_baseline})
predictions['Baseline RF'] = y_pred_baseline
print(f"✓ Baseline Accuracy: {acc_baseline:.4f}")

# ============================================
# 7. Results Summary
# ============================================
print("FINAL RESULTS & COMPARISON")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
results_df['Improvement'] = ((results_df['Accuracy'] - acc_baseline) * 100).round(2)

print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"BEST MODEL: {best_model_name}")
print(f"Test Accuracy: {best_accuracy:.4f}")
print(f"Improvement: +{results_df.iloc[0]['Improvement']:.2f}%")
print(f"{'='*80}")

# Detailed report
y_pred_best = predictions[best_model_name]

print(f"\n Classification Report ({best_model_name}):")

print(classification_report(y_test, y_pred_best, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_best)
print("\n Confusion Matrix:")
print(cm)

# ============================================
# 8. Visualizations
# ============================================
print("\n[VISUALIZATION] Creating comprehensive plots...")

fig = plt.figure(figsize=(20, 12))

# Plot 1: Model Comparison
ax1 = plt.subplot(2, 3, 1)
results_sorted = results_df.sort_values('Accuracy')
colors = ['#FF6B6B' if acc < acc_baseline else '#4ECDC4' 
          for acc in results_sorted['Accuracy']]
bars = ax1.barh(results_sorted['Model'], results_sorted['Accuracy'], color=colors)
ax1.axvline(x=acc_baseline, color='red', linestyle='--', label='Baseline', linewidth=2)
ax1.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, results_sorted['Accuracy'])):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{val:.4f}', va='center', fontsize=9)

# Plot 2: Improvement Bar Chart
ax2 = plt.subplot(2, 3, 2)
improvements = results_df[results_df['Model'] != 'Baseline RF'].sort_values('Improvement')
colors2 = ['#FF6B6B' if x < 0 else '#45B7D1' for x in improvements['Improvement']]
ax2.barh(improvements['Model'], improvements['Improvement'], color=colors2)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
ax2.set_title('Improvement over Baseline', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=le.classes_, yticklabels=le.classes_, cbar_kws={'label': 'Count'})
ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax3.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

# Plot 4: Feature Importance (if available)
ax4 = plt.subplot(2, 3, 4)
if hasattr(best_rf, 'feature_importances_'):
    importances = best_rf.feature_importances_[:k_best]
    indices = np.argsort(importances)[-15:]
    ax4.barh(range(len(indices)), importances[indices], color='coral')
    ax4.set_xlabel('Importance', fontsize=11, fontweight='bold')
    ax4.set_title('Top 15 Feature Importances (RF)', fontsize=13, fontweight='bold')
    ax4.set_yticks(range(len(indices)))
    ax4.set_yticklabels([top_features[i] if i < len(top_features) else f'F{i}' 
                         for i in indices], fontsize=8)
    ax4.grid(axis='x', alpha=0.3)

# Plot 5: Top Models Comparison
ax5 = plt.subplot(2, 3, 5)
top_5 = results_df.head(5)
colors5 = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_5)))
bars5 = ax5.bar(range(len(top_5)), top_5['Accuracy'], color=colors5, edgecolor='black', linewidth=1.5)
ax5.set_xticks(range(len(top_5)))
ax5.set_xticklabels(top_5['Model'], rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax5.set_title('Top 5 Models', fontsize=13, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for bar, val in zip(bars5, top_5['Accuracy']):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 6: Class Distribution
ax6 = plt.subplot(2, 3, 6)
class_counts = pd.Series(y_train).value_counts()
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# ============================================
# Add this section at the END of your multi-class training code
# After all the model training and evaluation
# ============================================

import pickle

print("SAVING MODELS AND PREPROCESSORS - MULTI-CLASS")

# 1. Save the best model
with open('multiclass_model.pkl', 'wb') as f:
    pickle.dump(trained_models[best_model_name], f)
print(f"✓ Saved: multiclass_model.pkl ({best_model_name})")

# 2. Save the scaler
with open('scaler_multiclass.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler_multiclass.pkl")

# 3. Save polynomial transformer
with open('poly_transformer_multiclass.pkl', 'wb') as f:
    pickle.dump(poly, f)
print("✓ Saved: poly_transformer_multiclass.pkl")

# 4. Save label encoder
with open('label_encoder_multiclass.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✓ Saved: label_encoder_multiclass.pkl")

# 5. Save the 8 key features used for polynomial
top_8_features = feature_scores.head(8)['feature'].values
with open('feature_names_multiclass.pkl', 'wb') as f:
    pickle.dump({
        'top_8_features': top_8_features.tolist(),
        'all_selected_features': top_features.tolist(),
        'k_best': k_best
    }, f)
print("✓ Saved: feature_names_multiclass.pkl")

# 6. Save feature selector (optional but useful)
with open('feature_selector_multiclass.pkl', 'wb') as f:
    pickle.dump({
        'feature_scores': feature_scores,
        'selector_f': selector_f,
        'selector_mi': selector_mi
    }, f)
print("✓ Saved: feature_selector_multiclass.pkl")

# 7. Save model performance metrics
model_metrics = {
    'best_model': best_model_name,
    'best_accuracy': best_accuracy,
    'all_results': results_df.to_dict('records'),
    'confusion_matrix': cm.tolist(),
    'classes': le.classes_.tolist()
}

with open('model_metrics_multiclass.pkl', 'wb') as f:
    pickle.dump(model_metrics, f)
print("✓ Saved: model_metrics_multiclass.pkl")

# ============================================
# Summary of saved files
# ============================================
print("\nEssential files for deployment:")
print("  1. multiclass_model.pkl          - Trained model")
print("  2. scaler_multiclass.pkl         - StandardScaler")
print("  3. poly_transformer_multiclass.pkl - PolynomialFeatures")
print("  4. label_encoder_multiclass.pkl  - LabelEncoder")
print("  5. feature_names_multiclass.pkl  - Feature names")
print("\nOptional files:")
print("  6. feature_selector_multiclass.pkl - Feature selection info")
print("  7. model_metrics_multiclass.pkl    - Performance metrics")
print("\nModel Details:")
print(f"  - Best Model: {best_model_name}")
print(f"  - Test Accuracy: {best_accuracy:.4f}")
print(f"  - Classes: {le.classes_.tolist()}")
print(f"  - Top 8 Features: {top_8_features.tolist()}")

# ============================================
# Create a deployment info file
# ============================================
deployment_info = f"""
MULTI-CLASS EXOPLANET CLASSIFICATION MODEL
==========================================

Model Information:
- Model Type: {best_model_name}
- Test Accuracy: {best_accuracy:.4f}
- Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Classes: {le.classes_.tolist()}

Required Input Features (8):
1. koi_fpflag_ss      - Stellar Eclipse Flag (0 or 1)
2. koi_fpflag_co      - Centroid Offset Flag (0 or 1)
3. koi_dikco_msky     - PRF Angular Offset from KIC (arcseconds)
4. koi_dicco_msky     - PRF Angular Offset OOT (arcseconds)
5. koi_smet_err2      - Stellar Metallicity Error (negative)
6. earth_similarity   - Earth Similarity Index (calculated)
7. log_snr           - Log Signal-to-Noise Ratio (calculated)
8. koi_count         - Number of Planets in System

Calculated Features (need raw inputs):
- earth_similarity: requires koi_prad, koi_teq
- log_snr: requires koi_model_snr

Pipeline Steps:
1. Collect 8 features (calculate earth_similarity and log_snr)
2. StandardScaler transformation
3. PolynomialFeatures (degree=2, interaction_only=True)
4. Model prediction

Files needed for deployment:
- multiclass_model.pkl
- scaler_multiclass.pkl
- poly_transformer_multiclass.pkl
- label_encoder_multiclass.pkl
- feature_names_multiclass.pkl
"""

with open('deployment_info_multiclass.txt', 'w') as f:
    f.write(deployment_info)
print("\n✓ Saved: deployment_info_multiclass.txt")

print("\n" + "="*80)
print("ALL FILES SAVED SUCCESSFULLY!")
print("="*80)
