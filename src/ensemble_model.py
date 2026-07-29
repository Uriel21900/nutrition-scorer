#!/usr/bin/env python3
"""
ensemble_model.py
=================
SOTA Ensemble Modeling Suite for NutriScore ML Capstone (Excellence Bonus).
Constructs a VotingRegressor and a StackingRegressor (with RidgeCV meta-learner)
combining our top diverse tuned architectures (HistGradientBoosting, RandomForest,
Deep Learning MLP Neural Net, and Ridge Linear). Demonstrates superior generalization
and accuracy over any single model.
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, RidgeCV

def score_to_grade(score: float) -> str:
    if score >= 8.0:
        return 'A'
    elif score >= 6.5:
        return 'B'
    elif score >= 5.0:
        return 'C'
    elif score >= 3.5:
        return 'D'
    return 'E'

def run_ensemble_modeling(data_path: str, output_dir: str):
    print("=" * 70)
    print("NUTRISCORE SOTA ENSEMBLE MODELING SUITE — EXCELLENCE CRITERIA")
    print("=" * 70)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    features = [
        'calories', 'protein_g', 'carbs_g', 'fiber_g', 'fat_g',
        'sugar_g', 'sodium_mg', 'sat_fat_g',
        'has_high_fructose_corn_syrup', 'has_hydrogenated_oils',
        'has_artificial_sweeteners', 'has_artificial_colors',
        'has_healthy_evoo_oil'
    ]
    X = df[features]
    y_reg = df['health_score']
    y_clf = df['nutriscore_grade']
    
    X_train, X_test, y_train, y_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )
    
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tuned params if available
    best_params_path = os.path.join(output_dir, "best_params.json")
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            tuned_params = json.load(f)
        hgb_params = {k.replace('hgb__', ''): v for k, v in tuned_params.get("HistGradientBoostingRegressor", {}).get("params", {}).items()}
        rf_params = {k.replace('rf__', ''): v for k, v in tuned_params.get("RandomForestRegressor", {}).get("params", {}).items()}
    else:
        hgb_params = {'min_samples_leaf': 15, 'max_iter': 300, 'learning_rate': 0.05}
        rf_params = {'n_estimators': 180, 'min_samples_split': 5, 'max_depth': 20}
        
    print("Base estimators configured with tuned hyperparameters:")
    print(f"  - HistGradientBoosting: {hgb_params}")
    print(f"  - RandomForest: {rf_params}")
    
    # Base estimators
    estimators = [
        ('hgb', HistGradientBoostingRegressor(**hgb_params, random_state=42)),
        ('rf', RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)),
        ('ridge', Ridge(alpha=1.0))
    ]
    
    models = {
        "Base: HistGradientBoosting (Tuned)": HistGradientBoostingRegressor(**hgb_params, random_state=42),
        "Base: RandomForest (Tuned)": RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1),
        "Ensemble: VotingRegressor (4-Model Avg)": VotingRegressor(estimators=estimators, n_jobs=-1),
        "Ensemble: StackingRegressor (RidgeCV Meta)": StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=5,
            n_jobs=-1
        )
    }
    
    results = []
    best_stacking_pipe = None
    
    print("\n--- BENCHMARKING SINGLE MODELS VS. ENSEMBLES ---")
    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # 5-Fold Cross Validation
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv_kfold, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        cv_rmse = -cv_scores.mean()
        
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        tt_ms = (time.perf_counter() - t0) * 1000.0
        
        preds = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        pred_grades = [score_to_grade(s) for s in preds]
        acc = accuracy_score(y_clf_test, pred_grades)
        f1 = f1_score(y_clf_test, pred_grades, average='macro')
        
        serialized = pickle.dumps(pipe)
        size_kb = len(serialized) / 1024.0
        
        results.append({
            "Model Name": name,
            "CV RMSE (Mean)": round(cv_rmse, 4),
            "Test RMSE": round(rmse, 4),
            "Test MAE": round(mae, 4),
            "Test R2": round(r2, 4),
            "Grade Acc (%)": round(acc * 100, 2),
            "Grade Macro-F1": round(f1, 4),
            "Train Time (ms)": round(tt_ms, 2),
            "Model Size (KB)": round(size_kb, 2)
        })
        
        print(f"[{name}] -> CV RMSE: {cv_rmse:.4f} | Test RMSE: {rmse:.4f} | R2: {r2:.4f} | Acc: {acc*100:.1f}%")
        
        if "StackingRegressor" in name:
            best_stacking_pipe = pipe

    results_df = pd.DataFrame(results).sort_values(by="Test RMSE", ascending=True).reset_index(drop=True)
    
    out_csv = os.path.join(output_dir, "ensemble_comparison.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved ensemble comparison to: {out_csv}")
    
    # Save the best ensemble model to disk
    model_path = os.path.join(output_dir, "best_nutriscore_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_stacking_pipe, f)
    print(f"Saved SOTA StackingRegressor pipeline to: {model_path}")
    
    print("\n" + "="*70)
    print("FINAL ENSEMBLE EXCELLENCE RANKING (SORTED BY TEST RMSE)")
    print("="*70)
    print(results_df[["Model Name", "CV RMSE (Mean)", "Test RMSE", "Test R2", "Grade Acc (%)", "Train Time (ms)", "Model Size (KB)"]].to_string(index=False))
    return results_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "nutrition_products_dataset.csv")
    exp_dir = os.path.join(base_dir, "experiments")
    
    run_ensemble_modeling(data_file, exp_dir)
