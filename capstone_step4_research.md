# Capstone Step 4: Survey Existing Research and Reproduce Available Solutions

## 1. Documented Summary of Research Papers

To build the foundation for my custom `nutrition-scorer` algorithm, I surveyed existing literature regarding nutritional profiling models (specifically the widely adopted Nutri-Score) and recent machine learning approaches used to predict food healthiness from ingredient data.

*   **Paper 1: "Leveraging Machine Learning for NOVA Classification and Nutri-Score Prediction" (2026)**
    *   *Summary:* This paper highlights the use of Extreme Gradient Boosting (XGBoost) and Natural Language Processing (DistilBERT) to automate the calculation of the Nutri-Score and the NOVA food processing classification. The authors demonstrate that while traditional Nutri-Score heavily relies on tabular macronutrient data (achieving high predictability via XGBoost), it often fails to account for the *degree of processing* found in the text-based ingredient list (which NLP models can capture). 
    *   *Link to Capstone:* My capstone addresses the exact gap identified in this research. While standard algorithms evaluate macronutrients, my custom app parses unstructured ingredient text (e.g., catching "Red 40", "aspartame", or "extra virgin olive oil") to dramatically alter the baseline score, combining tabular macro ratios with ingredient-quality penalties/bonuses.

*   **Paper 2: "Natural language processing and machine learning approaches for food categorization and nutrition quality prediction" (2023)**
    *   *Summary:* This study explored how large datasets like Open Food Facts can be used to predict nutritional quality scores. They found that relying strictly on the European Nutri-Score logic often incorrectly scores single-ingredient healthy fats (like Olive Oil) or diet beverages heavily laden with artificial sweeteners. 
    *   *Link to Capstone:* This directly supports the core logic of my `nutrition-scorer` app. In my app, I purposefully implemented exception handling for zero-calorie items and healthy fats to correct the biases inherent in traditional scoring models highlighted by this study.

## 2. Documented Available Code Solutions (Public Repositories)

In researching how others have implemented nutritional scoring programmatically, I found the following open-source solutions:

1.  **pyNutriScore** (`https://github.com/lemonhead94/pyNutriScore`): A standard Python package that implements the strict mathematical formula of the official European Nutri-Score. It takes calories, sugar, saturated fats, sodium, fruits/vegetables %, fiber, and protein to output a continuous score and a categorical letter grade (A-E).
2.  **Open Food Facts Python SDK** (`https://github.com/openfoodfacts/openfoodfacts-python`): The official Python wrapper for the database my app uses. Rather than calculating the score locally, this SDK allows developers to fetch the pre-calculated Nutri-Score directly from their servers.

## 3. Shared Conclusion & Analysis: Improving Upon the Baseline

**What I Learned:** 
Through reproducing the baseline Nutri-Score (see attached Jupyter Notebook), I learned that the standard algorithmic approach is highly rigid. It uses a "points" system that heavily penalizes fats and calories while sometimes ignoring the underlying *source* of those calories. 

**How My Capstone Improves the Current Work:**
The standard `pyNutriScore` algorithm gave a highly processed diet drink (Crystal Light) a perfect "A" score simply because it has zero calories, while giving Extra Virgin Olive Oil (Graza) a "C" or "D" due to its high fat content. 

My Capstone app fundamentally improves upon this baseline by introducing a **hybrid heuristic**:
1.  **Ingredient parsing:** My app detects artificial sweeteners and food dyes, heavily penalizing the diet drink.
2.  **Macronutrient Context:** My app detects the "healthy fat" status of EVOO and overrides the fat penalty to reward it as a superfood.
3.  **Net Carbs:** My app calculates *net carbs* (Carbs - Fiber) to accurately score products for modern dietary needs (like Keto/diabetic diets), which the standard Nutri-Score ignores entirely.
