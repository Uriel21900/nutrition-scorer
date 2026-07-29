#!/usr/bin/env python3
"""
generate_dataset.py
===================
Generates a realistic dataset of 5,000+ food products with macronutrient profiles,
ingredient flags, and target health scores (continuous 1.0-10.0) and NutriScore
grades (discrete A-E).

This synthetic dataset models real-world nutritional distributions across 10 major
dietary categories, incorporating domain-specific scoring logic from NutriScore's
expert algorithm (app.js) with real-world noise and variance.
"""

import os
import numpy as np
import pandas as pd

# Set reproducible seed
np.random.seed(42)

def generate_food_dataset(num_samples: int = 5000) -> pd.DataFrame:
    categories = [
        "Whole Foods & Produce",
        "Lean Proteins & Poultry",
        "Processed Snacks & Sweets",
        "Beverages & Drinks",
        "Whole Grains & Legumes",
        "Fast Food & Prepared Meals",
        "Dairy & Cheese",
        "Condiments & Sauces",
        "Superfoods & Healthy Oils",
        "Dietary Supplements & Shakes"
    ]
    
    data = []
    for idx in range(1, num_samples + 1):
        cat = np.random.choice(categories)
        
        # Category-specific macronutrient distributions
        if cat == "Whole Foods & Produce":
            name = f"Fresh Produce Item #{idx}"
            calories = np.random.uniform(25, 120)
            protein = np.random.uniform(0.5, 4.0)
            carbs = np.random.uniform(5.0, 25.0)
            fiber = np.random.uniform(2.0, 8.0)
            fat = np.random.uniform(0.1, 2.0)
            sugar = np.random.uniform(2.0, 15.0)
            sodium = np.random.uniform(5.0, 50.0)
            sat_fat = np.random.uniform(0.0, 0.5)
            has_hfcs = False
            has_hydro = False
            has_aspartame = False
            has_art_colors = False
            has_evoo = np.random.rand() < 0.1
            
        elif cat == "Lean Proteins & Poultry":
            name = f"Lean Protein #{idx}"
            calories = np.random.uniform(110, 250)
            protein = np.random.uniform(20.0, 45.0)
            carbs = np.random.uniform(0.0, 5.0)
            fiber = np.random.uniform(0.0, 1.0)
            fat = np.random.uniform(1.5, 12.0)
            sugar = np.random.uniform(0.0, 2.0)
            sodium = np.random.uniform(50.0, 450.0)
            sat_fat = np.random.uniform(0.5, 3.5)
            has_hfcs = False
            has_hydro = False
            has_aspartame = False
            has_art_colors = False
            has_evoo = np.random.rand() < 0.2
            
        elif cat == "Processed Snacks & Sweets":
            name = f"Processed Snack #{idx}"
            calories = np.random.uniform(200, 550)
            protein = np.random.uniform(1.0, 6.0)
            carbs = np.random.uniform(30.0, 75.0)
            fiber = np.random.uniform(0.5, 3.0)
            fat = np.random.uniform(8.0, 30.0)
            sugar = np.random.uniform(12.0, 45.0)
            sodium = np.random.uniform(150.0, 800.0)
            sat_fat = np.random.uniform(2.5, 12.0)
            has_hfcs = np.random.rand() < 0.45
            has_hydro = np.random.rand() < 0.40
            has_aspartame = np.random.rand() < 0.10
            has_art_colors = np.random.rand() < 0.60
            has_evoo = False
            
        elif cat == "Beverages & Drinks":
            name = f"Beverage Item #{idx}"
            calories = np.random.uniform(0, 240)
            protein = np.random.uniform(0.0, 2.0)
            carbs = np.random.uniform(0.0, 60.0)
            fiber = np.random.uniform(0.0, 0.5)
            fat = np.random.uniform(0.0, 1.0)
            sugar = np.random.uniform(0.0, 55.0)
            sodium = np.random.uniform(10.0, 120.0)
            sat_fat = 0.0
            has_hfcs = np.random.rand() < 0.50
            has_hydro = False
            has_aspartame = np.random.rand() < 0.35
            has_art_colors = np.random.rand() < 0.50
            has_evoo = False
            
        elif cat == "Whole Grains & Legumes":
            name = f"Grain/Legume #{idx}"
            calories = np.random.uniform(130, 350)
            protein = np.random.uniform(6.0, 18.0)
            carbs = np.random.uniform(25.0, 65.0)
            fiber = np.random.uniform(5.0, 16.0)
            fat = np.random.uniform(1.0, 5.0)
            sugar = np.random.uniform(0.5, 5.0)
            sodium = np.random.uniform(5.0, 200.0)
            sat_fat = np.random.uniform(0.2, 1.0)
            has_hfcs = False
            has_hydro = False
            has_aspartame = False
            has_art_colors = False
            has_evoo = np.random.rand() < 0.15
            
        elif cat == "Fast Food & Prepared Meals":
            name = f"Prepared Meal #{idx}"
            calories = np.random.uniform(350, 950)
            protein = np.random.uniform(12.0, 38.0)
            carbs = np.random.uniform(35.0, 95.0)
            fiber = np.random.uniform(1.5, 6.0)
            fat = np.random.uniform(14.0, 55.0)
            sugar = np.random.uniform(4.0, 22.0)
            sodium = np.random.uniform(450.0, 1800.0)
            sat_fat = np.random.uniform(4.0, 18.0)
            has_hfcs = np.random.rand() < 0.25
            has_hydro = np.random.rand() < 0.35
            has_aspartame = False
            has_art_colors = np.random.rand() < 0.30
            has_evoo = np.random.rand() < 0.10
            
        elif cat == "Dairy & Cheese":
            name = f"Dairy Product #{idx}"
            calories = np.random.uniform(80, 400)
            protein = np.random.uniform(6.0, 25.0)
            carbs = np.random.uniform(2.0, 18.0)
            fiber = 0.0
            fat = np.random.uniform(3.0, 32.0)
            sugar = np.random.uniform(2.0, 16.0)
            sodium = np.random.uniform(40.0, 650.0)
            sat_fat = np.random.uniform(2.0, 19.0)
            has_hfcs = np.random.rand() < 0.10
            has_hydro = False
            has_aspartame = np.random.rand() < 0.10
            has_art_colors = np.random.rand() < 0.15
            has_evoo = False
            
        elif cat == "Condiments & Sauces":
            name = f"Condiment #{idx}"
            calories = np.random.uniform(15, 180)
            protein = np.random.uniform(0.0, 2.0)
            carbs = np.random.uniform(2.0, 35.0)
            fiber = np.random.uniform(0.0, 1.5)
            fat = np.random.uniform(0.0, 18.0)
            sugar = np.random.uniform(1.0, 28.0)
            sodium = np.random.uniform(200.0, 1400.0)
            sat_fat = np.random.uniform(0.0, 3.0)
            has_hfcs = np.random.rand() < 0.55
            has_hydro = np.random.rand() < 0.20
            has_aspartame = np.random.rand() < 0.15
            has_art_colors = np.random.rand() < 0.40
            has_evoo = np.random.rand() < 0.25
            
        elif cat == "Superfoods & Healthy Oils":
            name = f"Superfood Oil/Nut #{idx}"
            calories = np.random.uniform(120, 280)
            protein = np.random.uniform(1.0, 9.0)
            carbs = np.random.uniform(0.0, 8.0)
            fiber = np.random.uniform(0.0, 5.0)
            fat = np.random.uniform(10.0, 28.0)
            sugar = np.random.uniform(0.0, 2.0)
            sodium = np.random.uniform(0.0, 50.0)
            sat_fat = np.random.uniform(1.0, 4.0)
            has_hfcs = False
            has_hydro = False
            has_aspartame = False
            has_art_colors = False
            has_evoo = True
            
        else:  # Dietary Supplements & Shakes
            name = f"Nutrition Shake/Supplement #{idx}"
            calories = np.random.uniform(100, 320)
            protein = np.random.uniform(15.0, 45.0)
            carbs = np.random.uniform(3.0, 25.0)
            fiber = np.random.uniform(1.0, 8.0)
            fat = np.random.uniform(1.5, 9.0)
            sugar = np.random.uniform(0.5, 12.0)
            sodium = np.random.uniform(50.0, 350.0)
            sat_fat = np.random.uniform(0.5, 2.5)
            has_hfcs = np.random.rand() < 0.10
            has_hydro = False
            has_aspartame = np.random.rand() < 0.40
            has_art_colors = np.random.rand() < 0.30
            has_evoo = False

        # Ensure physical consistency
        fiber = min(fiber, carbs)
        sugar = min(sugar, carbs)
        sat_fat = min(sat_fat, fat)
        
        net_carbs = max(0.0, carbs - fiber)
        calculated_cals = (protein * 4) + (net_carbs * 4) + (fat * 9)
        effective_cals = max(1.0, max(calories, calculated_cals))
        
        # Calculate domain base score (following app.js logic)
        base_score = 5.0
        
        protein_ratio = (protein * 4) / effective_cals
        carb_ratio = (net_carbs * 4) / effective_cals
        
        # Trivial Calories bypass
        bypass_carbs = False
        if calories < 30 and carbs < 5:
            bypass_carbs = True
            base_score += 2.0
            
        # Healthy fat / single-ingredient superfood bonus
        if protein_ratio < 0.05 and carb_ratio < 0.05 and fat > 0:
            if has_evoo:
                base_score += 3.5
            else:
                base_score += 1.5

        # Protein density scoring
        if protein_ratio >= 0.6:
            base_score += 4.0
        elif protein_ratio >= 0.4:
            base_score += 3.5
        elif protein_ratio >= 0.2:
            base_score += 2.0
        elif protein_ratio >= 0.1:
            base_score += 1.0

        # Fiber scoring
        fiber_per_100cal = (fiber / effective_cals) * 100
        if fiber_per_100cal >= 3.0:
            base_score += 2.0
        elif fiber_per_100cal >= 1.5:
            base_score += 1.0
        elif fiber_per_100cal < 0.5 and carb_ratio > 0.5 and not bypass_carbs:
            base_score -= 0.5

        # Carb / Sugar penalty
        if carb_ratio > 0.7:
            base_score -= 2.0
        elif carb_ratio > 0.5:
            base_score -= 1.0
            
        sugar_ratio = (sugar * 4) / effective_cals
        if sugar_ratio > 0.4:
            base_score -= 1.5

        # Sodium penalty
        sodium_per_cal = sodium / effective_cals
        if sodium_per_cal > 2.5:
            base_score -= 1.5
        elif sodium_per_cal > 1.5:
            base_score -= 0.8

        # Saturated fat penalty
        sat_fat_ratio = (sat_fat * 9) / effective_cals
        if sat_fat_ratio > 0.35:
            base_score -= 1.5
        elif sat_fat_ratio > 0.20:
            base_score -= 0.7

        # Ingredient penalties (bad ingredients from app.js)
        bad_count = sum([has_hfcs, has_hydro, has_aspartame, has_art_colors])
        base_score -= (bad_count * 1.5)

        # Add realistic non-linear nutritional interaction noise (standard normal noise ~0.35)
        # This models biological bioavailability differences and real-world variance
        noise = np.random.normal(0.0, 0.35)
        final_score = np.clip(base_score + noise, 1.0, 10.0)
        final_score = round(float(final_score), 2)

        # Discrete NutriScore Grade assignment (A, B, C, D, E)
        if final_score >= 8.0:
            grade = 'A'
        elif final_score >= 6.5:
            grade = 'B'
        elif final_score >= 5.0:
            grade = 'C'
        elif final_score >= 3.5:
            grade = 'D'
        else:
            grade = 'E'

        data.append({
            "product_id": f"PROD_{idx:05d}",
            "name": name,
            "category": cat,
            "calories": round(calories, 1),
            "protein_g": round(protein, 1),
            "carbs_g": round(carbs, 1),
            "fiber_g": round(fiber, 1),
            "fat_g": round(fat, 1),
            "sugar_g": round(sugar, 1),
            "sodium_mg": round(sodium, 1),
            "sat_fat_g": round(sat_fat, 1),
            "has_high_fructose_corn_syrup": int(has_hfcs),
            "has_hydrogenated_oils": int(has_hydro),
            "has_artificial_sweeteners": int(has_aspartame),
            "has_artificial_colors": int(has_art_colors),
            "has_healthy_evoo_oil": int(has_evoo),
            "health_score": final_score,
            "nutriscore_grade": grade
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "nutrition_products_dataset.csv")
    
    print("Generating synthetic nutrition dataset (5,000 products across 10 dietary categories)...")
    df = generate_food_dataset(5000)
    df.to_csv(out_path, index=False)
    print(f"Successfully saved dataset to: {out_path}")
    print(f"Dataset Shape: {df.shape}")
    print("\nScore Distribution:")
    print(df['health_score'].describe())
    print("\nGrade Distribution:")
    print(df['nutriscore_grade'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
