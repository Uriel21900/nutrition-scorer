# Step 4: Survey Existing Research — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 1**

---

## 1. Existing Literature & Standards
In developing NutriScore, we surveyed existing food classification standards and academic research (full research report available in `capstone_step4_research.md`):

1. **European Nutri-Score Standard (Rayner Score / FSA-NPS):**
   - Uses a point-subtraction algorithm based on energy, saturated fats, sugars, and sodium, offset by fiber, protein, and fruit/vegetable percentage.
   - *Limitation:* Purely additive rules fail to capture non-linear interactions between refined additives (e.g., HFCS combined with trans-fats).
2. **WHO Nutrient Profile Models:**
   - Establishes maximum thresholds for added sugars and sodium per food category.
3. **ML Approaches to Food Composition Analysis:**
   - Recent papers utilize Gradient Boosted Decision Trees (GBDT) and Multi-Layer Perceptrons (MLPs) to predict glycemic index and nutrient density from ingredient text embeddings.

---

## 2. Our Advancement Over Existing Approaches
- **Non-Linear Ensemble Modeling:** Instead of rigid heuristic tables, NutriScore trains a **StackingRegressor** combining `HistGradientBoostingRegressor`, `RandomForestRegressor`, `MLPRegressor`, and `Ridge` linear regression.
- **NLP Ingredient Flagging:** Automatically detects hazardous additives (e.g., artificial sweeteners, hydrogenated oils) and applies learned penalties that adapt to overall macronutrient density.
