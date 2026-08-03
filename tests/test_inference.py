"""
Unit tests for NutriScore ML Inference Engine
"""

import pytest
from src.api.inference import NutriScoreInferenceEngine, score_to_grade, parse_ingredients_flags


def test_score_to_grade_mapping():
    assert score_to_grade(9.0) == "A"
    assert score_to_grade(8.0) == "A"
    assert score_to_grade(7.2) == "B"
    assert score_to_grade(6.5) == "B"
    assert score_to_grade(5.5) == "C"
    assert score_to_grade(5.0) == "C"
    assert score_to_grade(4.0) == "D"
    assert score_to_grade(3.5) == "D"
    assert score_to_grade(2.0) == "E"
    assert score_to_grade(1.0) == "E"


def test_parse_ingredients_flags():
    flags = parse_ingredients_flags("Water, High Fructose Corn Syrup, Aspartame, Red 40, Extra Virgin Olive Oil")
    assert flags["has_high_fructose_corn_syrup"] == 1
    assert flags["has_artificial_sweeteners"] == 1
    assert flags["has_artificial_colors"] == 1
    assert flags["has_healthy_evoo_oil"] == 1
    assert flags["has_hydrogenated_oils"] == 0


def test_inference_engine_prediction_healthy_item():
    engine = NutriScoreInferenceEngine()
    input_payload = {
        "calories": 180,
        "protein_g": 22.0,
        "carbs_g": 10.0,
        "fiber_g": 5.0,
        "fat_g": 4.0,
        "sugar_g": 1.0,
        "sodium_mg": 120.0,
        "ingredients": "Organic Salmon, Olive Oil, Salt"
    }
    result = engine.predict(input_payload)
    assert "score" in result
    assert "grade" in result
    assert "insights" in result
    assert 1.0 <= result["score"] <= 10.0
    assert result["grade"] in ["A", "B", "C", "D", "E"]
    # Should be high scoring item
    assert result["score"] >= 6.5


def test_inference_engine_prediction_unhealthy_item():
    engine = NutriScoreInferenceEngine()
    input_payload = {
        "calories": 350,
        "protein_g": 1.0,
        "carbs_g": 65.0,
        "fiber_g": 0.0,
        "fat_g": 15.0,
        "sugar_g": 45.0,
        "sodium_mg": 450.0,
        "ingredients": "Sugar, High Fructose Corn Syrup, Hydrogenated Palm Oil, Red 40, Yellow 5"
    }
    result = engine.predict(input_payload)
    assert "score" in result
    assert "grade" in result
    assert 1.0 <= result["score"] <= 10.0
    # Should be low scoring item (D or E)
    assert result["grade"] in ["D", "E"]
    assert len(result["insights"]["negative_factors"]) > 0


def test_inference_engine_boundary_values():
    engine = NutriScoreInferenceEngine()
    input_payload = {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fiber_g": 0.0,
        "fat_g": 0.0,
        "sugar_g": 0.0,
        "sodium_mg": 0.0,
        "ingredients": ""
    }
    result = engine.predict(input_payload)
    assert 1.0 <= result["score"] <= 10.0
    assert result["grade"] in ["A", "B", "C", "D", "E"]
