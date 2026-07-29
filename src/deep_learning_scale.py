#!/usr/bin/env python3
"""
deep_learning_scale.py
======================
Implements Large-Scale Deep Learning using PyTorch (nn.Module, DataLoader,
BatchNorm, Dropout, and learning rate scheduling) for NutriScore Step 8.

Demonstrates GPU/CPU scalable neural network training capable of handling
large datasets with high throughput and low generalization error.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class NutriScoreDataset(Dataset):
    """Custom PyTorch Dataset for high-performance mini-batch loading."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class NutriScoreDNN(nn.Module):
    """
    4-Layer Deep Neural Network architecture for scalable food health score prediction.
    Includes Batch Normalization and Dropout for regularizing large-scale data.
    """
    def __init__(self, input_dim: int = 13):
        super(NutriScoreDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def run_deep_learning_experiment(data_dir: str, exp_dir: str):
    print("=" * 70)
    print("NUTRISCORE STEP 8: LARGE-SCALE DEEP LEARNING (PYTORCH DNN)")
    print("=" * 70)
    
    os.makedirs(exp_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "nutrition_products_dataset.csv")
    df = pd.read_csv(csv_path)
    
    features = [
        'calories', 'protein_g', 'carbs_g', 'fiber_g', 'fat_g',
        'sugar_g', 'sodium_mg', 'sat_fat_g',
        'has_high_fructose_corn_syrup', 'has_hydrogenated_oils',
        'has_artificial_sweeteners', 'has_artificial_colors',
        'has_healthy_evoo_oil'
    ]
    target = 'health_score'
    
    X = df[features].values
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = NutriScoreDataset(X_train_scaled, y_train)
    test_dataset = NutriScoreDataset(X_test_scaled, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch DL] Training on Device: {device} | Batch Size: {batch_size}")
    
    model = NutriScoreDNN(input_dim=len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 40
    start_time = time.time()
    loss_history = []
    
    print("[PyTorch DL] Training Deep Neural Network across batches...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  -> Epoch [{epoch:2d}/{epochs}] - Train MSE: {epoch_loss:.4f} - RMSE: {np.sqrt(epoch_loss):.4f}")
            
    train_duration = time.time() - start_time
    total_samples = len(train_dataset) * epochs
    throughput_sps = total_samples / max(train_duration, 0.001)
    
    # Evaluation
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.append(out.cpu().numpy())
            
    y_pred = np.vstack(preds).flatten()
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2 = float(r2_score(y_test, y_pred))
    
    model_path = os.path.join(exp_dir, "nutriscore_dnn.pth")
    torch.save(model.state_dict(), model_path)
    model_size_kb = os.path.getsize(model_path) / 1024.0
    
    print("\n--- PyTorch Deep Learning Scaling Results ---")
    print(f"  -> Total Samples Processed: {total_samples:,} across {epochs} epochs")
    print(f"  -> Training Duration:       {train_duration:.2f} seconds")
    print(f"  -> Training Throughput:     {throughput_sps:,.0f} samples / sec")
    print(f"  -> PyTorch DL Test RMSE:    {test_rmse:.4f}")
    print(f"  -> PyTorch DL Test R²:      {test_r2:.4f}")
    print(f"  -> Serialized Model Size:   {model_size_kb:.2f} KB ({model_path})")
    
    results = {
        "paradigm": "PyTorch Deep Neural Network (DataLoader + BatchNorm + Dropout)",
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "total_samples_processed": total_samples,
        "training_time_sec": round(train_duration, 3),
        "throughput_samples_per_sec": round(throughput_sps, 1),
        "test_rmse": round(test_rmse, 4),
        "test_r2": round(test_r2, 4),
        "model_size_kb": round(model_size_kb, 2),
        "loss_history_rmse": [round(float(np.sqrt(val)), 4) for val in loss_history]
    }
    
    out_file = os.path.join(exp_dir, "deep_learning_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Successfully saved PyTorch DL benchmark to: {out_file}\n")
    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    exp_dir = os.path.join(base_dir, "experiments")
    
    run_deep_learning_experiment(data_dir, exp_dir)
