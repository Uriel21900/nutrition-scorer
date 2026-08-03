"""
NutriScore ML Inference Engine & Open Food Facts Integration
===========================================================
Loads trained ensemble models (StackingRegressor / VotingRegressor) and provides
production-ready prediction methods with schema validation, fallback logic, and
feature breakdown analytics.
"""

import os
import pickle
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from ..monitoring.logger import get_logger

logger = get_logger("NutriScoreInference")

FEATURES = [
    'calories', 'protein_g', 'carbs_g', 'fiber_g', 'fat_g',
    'sugar_g', 'sodium_mg', 'sat_fat_g',
    'has_high_fructose_corn_syrup', 'has_hydrogenated_oils',
    'has_artificial_sweeteners', 'has_artificial_colors',
    'has_healthy_evoo_oil'
]


def score_to_grade(score: float) -> str:
    """
    Maps continuous NutriScore health score (1.0 - 10.0) to A-E Letter Grades.
    """
    if score >= 8.0:
        return 'A'
    elif score >= 6.5:
        return 'B'
    elif score >= 5.0:
        return 'C'
    elif score >= 3.5:
        return 'D'
    return 'E'


def parse_ingredients_flags(ingredients_text: str) -> Dict[str, int]:
    """
    Parses ingredient text strings into binary flags for model features.
    """
    text = (ingredients_text or "").lower()
    return {
        "has_high_fructose_corn_syrup": 1 if ("high fructose" in text or "hfcs" in text) else 0,
        "has_hydrogenated_oils": 1 if ("hydrogenated" in text or "palm oil" in text) else 0,
        "has_artificial_sweeteners": 1 if any(w in text for w in ["aspartame", "sucralose", "saccharin", "acesulfame"]) else 0,
        "has_artificial_colors": 1 if any(w in text for w in ["red 40", "yellow 5", "yellow 6", "blue 1", "titanium dioxide"]) else 0,
        "has_healthy_evoo_oil": 1 if any(w in text for w in ["extra virgin olive oil", "avocado oil", "olive oil"]) else 0,
    }


class NutriScoreInferenceEngine:
    """
    Production inference engine for NutriScore ML models.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.logger = get_logger("NutriScoreInference")
        self.model = None
        self.model_name = "AnalyticalHeuristicFallback-v1.0"
        self.is_ml_model = False

        if model_path is None:
            # Look in experiments/ directory relative to root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "experiments", "best_nutriscore_model.pkl")

        self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Loads pickled ML model pipeline from disk.
        """
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_name = os.path.basename(model_path)
                self.is_ml_model = True
                self.logger.info(f"Loaded ML model successfully from {model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load pickle model {model_path}: {e}")
        else:
            self.logger.warning(f"Model path not found: {model_path}. Using fallback analytical model.")
        
        self.model = None
        self.is_ml_model = False
        return False

    def _analytical_heuristic_score(self, features: Dict[str, float]) -> float:
        """
        Analytical domain-knowledge heuristic scoring algorithm (1.0 - 10.0 scale)
        used as fallback or explanation baseline.
        """
        calories = max(features.get("calories", 0.0), 1.0)
        protein = features.get("protein_g", 0.0)
        carbs = features.get("carbs_g", 0.0)
        fiber = features.get("fiber_g", 0.0)
        fat = features.get("fat_g", 0.0)
        sugar = features.get("sugar_g", 0.0)
        sodium = features.get("sodium_mg", 0.0)

        base_score = 5.0
        
        # Protein density bonus
        protein_ratio = (protein * 4) / calories
        if protein_ratio >= 0.4:
            base_score += 3.0
        elif protein_ratio >= 0.2:
            base_score += 1.5

        # Fiber density bonus
        fiber_ratio = (fiber * 4) / calories
        if fiber_ratio >= 0.15:
            base_score += 2.0
        elif fiber_ratio >= 0.05:
            base_score += 1.0

        # Sugar penalty
        sugar_ratio = (sugar * 4) / calories
        if sugar_ratio > 0.4:
            base_score -= 2.5
        elif sugar_ratio > 0.2:
            base_score -= 1.0

        # Sodium penalty (per 100 kcal)
        sodium_per_100kcal = (sodium / calories) * 100
        if sodium_per_100kcal > 400:
            base_score -= 2.0
        elif sodium_per_100kcal > 200:
            base_score -= 1.0

        # Ingredient flags penalty / bonus
        base_score -= 1.5 * features.get("has_high_fructose_corn_syrup", 0)
        base_score -= 1.5 * features.get("has_hydrogenated_oils", 0)
        base_score -= 1.0 * features.get("has_artificial_sweeteners", 0)
        base_score -= 0.8 * features.get("has_artificial_colors", 0)
        base_score += 1.5 * features.get("has_healthy_evoo_oil", 0)

        return float(np.clip(base_score, 1.0, 10.0))

    def prepare_feature_vector(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Normalizes input keys (supporting both camelCase and snake_case) and
        returns a 1-row DataFrame matching the 13 feature schema.
        """
        ingredients_text = str(input_data.get("ingredients", input_data.get("ingredients_text", "")))
        flags = parse_ingredients_flags(ingredients_text)

        row = {
            "calories": float(input_data.get("calories", 100.0)),
            "protein_g": float(input_data.get("protein_g", input_data.get("protein", 0.0))),
            "carbs_g": float(input_data.get("carbs_g", input_data.get("carbs", 0.0))),
            "fiber_g": float(input_data.get("fiber_g", input_data.get("fiber", 0.0))),
            "fat_g": float(input_data.get("fat_g", input_data.get("fat", 0.0))),
            "sugar_g": float(input_data.get("sugar_g", input_data.get("sugar", 0.0))),
            "sodium_mg": float(input_data.get("sodium_mg", input_data.get("sodium", 0.0))),
            "sat_fat_g": float(input_data.get("sat_fat_g", input_data.get("sat_fat", 0.0))),
            "has_high_fructose_corn_syrup": flags["has_high_fructose_corn_syrup"],
            "has_hydrogenated_oils": flags["has_hydrogenated_oils"],
            "has_artificial_sweeteners": flags["has_artificial_sweeteners"],
            "has_artificial_colors": flags["has_artificial_colors"],
            "has_healthy_evoo_oil": flags["has_healthy_evoo_oil"],
        }
        return pd.DataFrame([row], columns=FEATURES)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes prediction inference and returns structured grade, score, and breakdown.
        """
        df_feat = self.prepare_feature_vector(input_data)
        feat_dict = df_feat.iloc[0].to_dict()

        if self.is_ml_model and self.model is not None:
            try:
                raw_pred = float(self.model.predict(df_feat)[0])
                score = float(np.clip(round(raw_pred, 2), 1.0, 10.0))
                method = self.model_name
            except Exception as e:
                self.logger.error(f"ML model prediction error ({e}); falling back to heuristic.")
                score = round(self._analytical_heuristic_score(feat_dict), 2)
                method = "AnalyticalHeuristicFallback-v1.0"
        else:
            score = round(self._analytical_heuristic_score(feat_dict), 2)
            method = "AnalyticalHeuristicFallback-v1.0"

        grade = score_to_grade(score)

        # Feature contribution analytics
        positive_factors = []
        negative_factors = []
        if feat_dict["protein_g"] >= 10.0:
            positive_factors.append("High Protein Density")
        if feat_dict["fiber_g"] >= 3.0:
            positive_factors.append("Rich in Dietary Fiber")
        if feat_dict["has_healthy_evoo_oil"]:
            positive_factors.append("Contains Heart-Healthy Olive/Avocado Oil")

        if feat_dict["sugar_g"] >= 15.0:
            negative_factors.append("High Added Sugar Content")
        if feat_dict["sodium_mg"] >= 400.0:
            negative_factors.append("High Sodium Content")
        if feat_dict["has_high_fructose_corn_syrup"]:
            negative_factors.append("Contains High Fructose Corn Syrup")
        if feat_dict["has_hydrogenated_oils"]:
            negative_factors.append("Contains Hydrogenated / Palm Oils")
        if feat_dict["has_artificial_sweeteners"]:
            negative_factors.append("Contains Artificial Sweeteners")

        return {
            "score": score,
            "grade": grade,
            "model_used": method,
            "features_analyzed": feat_dict,
            "insights": {
                "positive_factors": positive_factors,
                "negative_factors": negative_factors,
                "confidence_interval": [round(max(1.0, score - 0.35), 2), round(min(10.0, score + 0.35), 2)]
            }
        }

    def predict_from_open_food_facts(self, barcode: str) -> Dict[str, Any]:
        """
        Fetches product metadata from Open Food Facts API and returns prediction.
        """
        clean_barcode = "".join(ch for ch in str(barcode) if ch.isdigit())
        url = f"https://world.openfoodfacts.org/api/v0/product/{clean_barcode}.json"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NutriScore-Capstone-MLE/1.0 (https://github.com/Uriel21900/nutrition-scorer)"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            if data.get("status") != 1 or "product" not in data:
                return {
                    "success": False,
                    "error": f"Barcode {barcode} not found in Open Food Facts database."
                }

            prod = data["product"]
            nutr = prod.get("nutriments", {})

            product_data = {
                "product_name": prod.get("product_name", "Unknown Product"),
                "brand": prod.get("brands", "Unknown Brand"),
                "calories": float(nutr.get("energy-kcal_100g", nutr.get("energy-kcal", 150.0)) or 150.0),
                "protein_g": float(nutr.get("proteins_100g", nutr.get("proteins", 0.0)) or 0.0),
                "carbs_g": float(nutr.get("carbohydrates_100g", nutr.get("carbohydrates", 15.0)) or 0.0),
                "fiber_g": float(nutr.get("fiber_100g", nutr.get("fiber", 0.0)) or 0.0),
                "fat_g": float(nutr.get("fat_100g", nutr.get("fat", 5.0)) or 0.0),
                "sugar_g": float(nutr.get("sugars_100g", nutr.get("sugars", 0.0)) or 0.0),
                "sodium_mg": float(nutr.get("sodium_100g", 0.0) * 1000 or nutr.get("salt_100g", 0.0) * 400 or 100.0),
                "sat_fat_g": float(nutr.get("saturated-fat_100g", nutr.get("saturated-fat", 1.0)) or 0.0),
                "ingredients": prod.get("ingredients_text_en", prod.get("ingredients_text", ""))
            }

            prediction_res = self.predict(product_data)
            prediction_res["success"] = True
            prediction_res["barcode"] = clean_barcode
            prediction_res["product_metadata"] = {
                "name": product_data["product_name"],
                "brand": product_data["brand"],
                "image_url": prod.get("image_front_small_url", prod.get("image_url", ""))
            }
            return prediction_res

        except Exception as e:
            self.logger.error(f"Open Food Facts lookup error for barcode {barcode}: {e}")
            return {
                "success": False,
                "error": f"Failed to retrieve Open Food Facts data: {str(e)}"
            }
