"""
Unit tests for all modules in the epic-discount-predictor project.
Run with: pytest tests/test_modules.py -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────
# cleaner.py tests
# ─────────────────────────────────────────

from modules.cleaner import (
    clean_price,
    normalize_title,
    extract_discount_percent,
    clean_deals_dataframe,
)


def test_clean_price_with_dollar_sign():
    """Test that clean_price correctly extracts float from $19.99."""
    assert clean_price("$19.99") == 19.99


def test_clean_price_with_usd_suffix():
    """Test that clean_price handles '29.99 USD' format."""
    assert clean_price("29.99 USD") == 29.99


def test_clean_price_with_none():
    """Test that clean_price returns None for None input."""
    assert clean_price(None) is None


def test_clean_price_with_integer_string():
    """Test that clean_price handles integer price strings."""
    assert clean_price("10") == 10.0


def test_normalize_title_removes_special_chars():
    """Test that normalize_title strips punctuation."""
    assert normalize_title("STAR WARS: Jedi!") == "star wars jedi"


def test_normalize_title_lowercases():
    """Test that normalize_title returns lowercase string."""
    assert normalize_title("Elden Ring") == "elden ring"


def test_normalize_title_empty_string():
    """Test that normalize_title handles empty string."""
    assert normalize_title("") == ""


def test_normalize_title_none():
    """Test that normalize_title handles None input."""
    assert normalize_title(None) == ""


def test_extract_discount_percent_basic():
    """Test extraction of discount from 'Save 75% today'."""
    assert extract_discount_percent("Save 75% today") == 75.0


def test_extract_discount_percent_no_match():
    """Test that extract_discount_percent returns None if no % found."""
    assert extract_discount_percent("No discount here") is None


def test_extract_discount_percent_none():
    """Test that extract_discount_percent handles None input."""
    assert extract_discount_percent(None) is None


def test_clean_deals_dataframe_empty():
    """Test that clean_deals_dataframe handles empty DataFrame."""
    result = clean_deals_dataframe(pd.DataFrame())
    assert result.empty


def test_clean_deals_dataframe_adds_columns():
    """Test that clean_deals_dataframe adds expected feature columns."""
    df = pd.DataFrame({
        "title": ["Elden Ring"],
        "salePrice": ["39.99"],
        "normalPrice": ["59.99"],
        "savings": ["33.33"],
        "metacriticScore": ["94"],
        "isOnSale": ["1"],
        "releaseDate": [1645000000],
        "lastChange": [1700000000],
    })
    result = clean_deals_dataframe(df)
    assert "title_clean" in result.columns
    assert "is_on_sale" in result.columns
    assert "game_age_days" in result.columns
    assert "days_since_last_change" in result.columns
    assert "discount_depth" in result.columns


# ─────────────────────────────────────────
# scraper.py tests
# ─────────────────────────────────────────

from modules.scraper import extract_score_from_text


def test_extract_score_from_text_fraction():
    """Test score extraction from '85/100' format."""
    assert extract_score_from_text("Score: 85/100") == 85.0


def test_extract_score_from_text_decimal():
    """Test score extraction from decimal format like '8.5'."""
    result = extract_score_from_text("Rating: 8.5")
    assert result == 8.5


def test_extract_score_from_text_none():
    """Test that extract_score_from_text handles None input."""
    assert extract_score_from_text(None) is None


def test_extract_score_from_text_no_score():
    """Test that extract_score_from_text returns None if no score."""
    assert extract_score_from_text("No score here") is None


# ─────────────────────────────────────────
# epic_api.py tests
# ─────────────────────────────────────────

from modules.epic_api import extract_game_info


def test_extract_game_info_basic():
    """Test that extract_game_info returns expected keys."""
    mock_game = {
        "id": "abc123",
        "title": "Test Game",
        "description": "A test game.",
        "status": "ACTIVE",
        "effectiveDate": "2024-01-01",
        "expiryDate": "2024-01-08",
        "price": {
            "totalPrice": {
                "originalPrice": 1999,
                "discountPrice": 0,
                "discount": 1999,
                "currencyCode": "USD",
            }
        },
        "seller": {"name": "Test Seller"},
        "promotions": None,
    }
    result = extract_game_info(mock_game)
    assert result["title"] == "Test Game"
    assert result["is_free"] is True
    assert result["currency"] == "USD"


def test_extract_game_info_no_promotions():
    """Test extract_game_info when promotions field is None."""
    mock_game = {
        "id": "xyz",
        "title": "No Promo Game",
        "description": "",
        "status": "ACTIVE",
        "effectiveDate": None,
        "expiryDate": None,
        "price": {"totalPrice": {
            "originalPrice": 0,
            "discountPrice": 0,
            "discount": 0,
            "currencyCode": "USD",
        }},
        "seller": {"name": "Dev"},
        "promotions": None,
    }
    result = extract_game_info(mock_game)
    assert result["has_current_promo"] is False
    assert result["has_upcoming_promo"] is False


# ─────────────────────────────────────────
# analysis.py tests
# ─────────────────────────────────────────

from modules.analysis import (
    get_feature_matrix,
    predict_sale_likelihood,
)


def test_get_feature_matrix_returns_correct_shape():
    """Test that get_feature_matrix returns correct X and y shapes."""
    df = pd.DataFrame({
        "normal_price": [59.99, 29.99, 19.99],
        "savings": [75.0, 50.0, 25.0],
        "metacritic_score": [90.0, 75.0, 60.0],
        "game_age_days": [500, 200, 100],
        "days_since_last_change": [30, 15, 5],
        "discount_depth": [75.0, 50.0, 25.0],
        "is_on_sale": [1, 1, 0],
    })
    X, y = get_feature_matrix(df)
    assert X.shape[0] == 3
    assert len(y) == 3


def test_get_feature_matrix_drops_nulls():
    """Test that get_feature_matrix drops rows with null values."""
    df = pd.DataFrame({
        "normal_price": [59.99, None, 19.99],
        "savings": [75.0, 50.0, 25.0],
        "metacritic_score": [90.0, 75.0, 60.0],
        "game_age_days": [500, 200, 100],
        "days_since_last_change": [30, 15, 5],
        "discount_depth": [75.0, 50.0, 25.0],
        "is_on_sale": [1, 1, 0],
    })
    X, y = get_feature_matrix(df)
    assert X.shape[0] == 2


def test_predict_sale_likelihood_returns_float():
    """Test that predict_sale_likelihood returns a float between 0 and 1."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 6))

    game = {
        "normal_price": 59.99,
        "savings": 0.0,
        "metacritic_score": 85.0,
        "game_age_days": 365,
        "days_since_last_change": 60,
        "discount_depth": 0.0,
    }
    result = predict_sale_likelihood(game, mock_model, mock_scaler)
    assert 0.0 <= result <= 1.0