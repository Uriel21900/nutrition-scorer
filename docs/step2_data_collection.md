# Step 2: Data Collection — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 1**

---

## 1. Data Acquisition Strategy
NutriScore leverages two primary data sources to train and benchmark our machine learning models:
1. **The Open Food Facts Database:** The world's largest open-source food products catalog containing over 2 million products with verified barcode numbers, ingredient lists, and standardized nutrient per 100g values.
2. **Curated & Synthesized Nutrition Datasets (`data/nutrition_products_dataset.csv`):** A clean dataset generated via `data/generate_dataset.py` representing 5,000+ realistic packaged food profiles across 10 food categories.

---

## 2. Feature Schema & Relevance to Problem
Our dataset includes **18 structured features and targets**, well-chosen to reflect nutritional quality:

| Feature Name | Type | Unit / Scale | Relevance |
| :--- | :--- | :--- | :--- |
| `calories` | Float | kcal / 100g | Baseline energy density |
| `protein_g` | Float | grams / 100g | Essential macro; positive health correlation |
| `carbs_g` | Float | grams / 100g | Carbohydrate load |
| `fiber_g` | Float | grams / 100g | Digestive health; offsets glycemic spike |
| `fat_g` | Float | grams / 100g | Total fat content |
| `sugar_g` | Float | grams / 100g | Added and natural sugar density (negative factor) |
| `sodium_mg` | Float | mg / 100g | Cardiovascular health penalty |
| `sat_fat_g` | Float | grams / 100g | Saturated fat penalty |
| `has_high_fructose_corn_syrup` | Binary | 0 or 1 | Hazardous refined sweetener flag |
| `has_hydrogenated_oils` | Binary | 0 or 1 | Trans-fat / hydrogenated oil flag |
| `has_artificial_sweeteners` | Binary | 0 or 1 | Chemical sweetener flag |
| `has_artificial_colors` | Binary | 0 or 1 | Synthetic dye flag |
| `has_healthy_evoo_oil` | Binary | 0 or 1 | Single-ingredient superfood oil bonus |
| **`health_score` (Target)** | Float | 1.0 – 10.0 | Continuous health score for Regression |
| **`nutriscore_grade` (Target)** | String | `A` – `E` | Discrete Letter Grade for Classification |

---

## 3. Data Cleaning & Integrity
- All missing numeric nutrient values are imputed using median category imputation.
- Outliers (e.g., negative macronutrient values or calories > 900 per 100g) are filtered.
- Text ingredient flags are parsed using lowercase keyword matching (`parse_ingredients_flags`).
