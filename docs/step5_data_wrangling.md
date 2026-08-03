# Step 5: Data Wrangling & Preprocessing — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 1**

---

## 1. Overview of Data Wrangling (`data_wrangling.ipynb`)
Our complete interactive data wrangling workflow is implemented in `data_wrangling.ipynb` and automated in `data/generate_dataset.py`.

### Key Preprocessing Steps:
1. **Schema Standardization:** Normalizing Open Food Facts JSON payloads into 13 standardized numeric and binary feature columns.
2. **Macronutrient Ratio Calculation:** Computing effective calories and verifying macro energy consistency ($4 \times \text{Protein} + 4 \times \text{NetCarbs} + 9 \times \text{Fat}$).
3. **Outlier Filtering:**
   - Removing products with negative macronutrient values.
   - Removing entries where sum of macronutrients exceeds 100g per 100g serving.
4. **Feature Scaling & Transformation:**
   - Applying `StandardScaler` within our `scikit-learn` Machine Learning Pipelines (`src/experiment_pipeline.py`) to prevent data leakage across cross-validation folds.
   - Binary additive flags (`0` or `1`) are kept unscaled for tree-based models and scaled for linear/MLP architectures.

---

## 2. Generated Dataset Verification
Running `python data/generate_dataset.py` produces `data/nutrition_products_dataset.csv`:
- **Sample Count:** 5,000+ verified food profiles.
- **Class Balance:** Stratified distribution across grades `A` (20%), `B` (22%), `C` (25%), `D` (18%), and `E` (15%).
- **No Missing Values:** Clean, complete tabular schema ready for automated ML benchmarking.
