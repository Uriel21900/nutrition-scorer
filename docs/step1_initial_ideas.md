# Step 1: Initial Project Ideas — NutriScore
**Machine Learning Engineering & AI Bootcamp Capstone — Phase 1**

---

## 1. Executive Summary & Project Selection
NutriScore is an intelligent food nutrition evaluation and classification system designed to provide an instant, objective health score (1.0 to 10.0) and letter grade (`A` through `E`) for food products from nutrition facts and ingredients lists.

### Selected Idea: Smart Nutrition Analyzer (NutriScore)
In evaluating practical machine learning problems for consumer health and food informatics, three primary concepts were explored:
1. **Recipe Price & Substitution Predictor:** Predicting ingredient cost fluctuations.
2. **NutriScore — Real-Time Food Quality & Grade Predictor (Selected):** A machine learning system that ingests macro ratios and ingredient text to score food products and detect unhealthy hidden additives.
3. **Personalized Dietary Allergen Alert API:** Flagging allergen risks from OCR scans.

**Why NutriScore Was Chosen:**
- **High Practical Application:** Consumers often struggle to decipher complex nutritional tables and ingredient names (e.g., distinguishing between natural sugars and High Fructose Corn Syrup, or healthy olive oil versus hydrogenated palm oils).
- **Justified Value for Client:** Grocery shoppers, fitness apps, and health monitoring platforms require an instant API that converts raw nutrition telemetry into an interpretable grade (`A`–`E`) and continuous score (1.0–10.0).
- **Appropriate Scope:** Involves both tabular regression/classification and NLP text flag parsing, scaling to web-scale datasets (Open Food Facts), and deploying as an interactive web app and REST API.

---

## 2. Problem Statement & Target Audience
- **The Problem:** Modern packaged foods use deceptive marketing and obscure ingredient naming to hide poor nutritional value.
- **The Solution:** An automated ML scoring engine that penalizes added sugars, sodium, and hazardous additives while rewarding lean protein, dietary fiber, and healthy single-ingredient fats.
- **Target Audience:** Consumers scanning barcodes in grocery stores, nutritionists auditing dietary intakes, and health apps integrating a scoring API.
