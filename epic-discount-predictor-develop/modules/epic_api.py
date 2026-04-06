"""
Module for fetching free and discounted game data
from the Epic Games Store promotional API.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("EPIC_BASE_URL")


def get_epic_promotions(locale="en-US", country="US"):
    """
    Fetch current and upcoming free game promotions from Epic Games Store.

    Args:
        locale (str): Locale string for language. Default is 'en-US'.
        country (str): Country code for regional pricing. Default is 'US'.

    Returns:
        list: A list of game dictionaries with promotion data,
        or empty list if request fails.
    """
    endpoint = f"{BASE_URL}/freeGamesPromotions"
    params = {
        "locale": locale,
        "country": country,
        "allowCountries": country,
    }
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        games = (
            data
            .get("data", {})
            .get("Catalog", {})
            .get("searchStore", {})
            .get("elements", [])
        )
        return games
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Epic promotions: {e}")
        return []


def extract_game_info(game):
    """
    Extract relevant fields from a raw Epic Games game dictionary.

    Args:
        game (dict): Raw game dictionary from Epic Games API.

    Returns:
        dict: Cleaned dictionary with key game fields extracted.
    """
    price_info = game.get("price", {}).get("totalPrice", {})
    promotions = game.get("promotions") or {}
    current_promos = promotions.get("promotionalOffers", [])
    upcoming_promos = promotions.get("upcomingPromotionalOffers", [])

    return {
        "id": game.get("id"),
        "title": game.get("title"),
        "description": game.get("description"),
        "status": game.get("status"),
        "effective_date": game.get("effectiveDate"),
        "original_price": price_info.get("originalPrice"),
        "discount_price": price_info.get("discountPrice"),
        "discount_amount": price_info.get("discount"),
        "currency": price_info.get("currencyCode"),
        "seller": game.get("seller", {}).get("name"),
        "is_free": price_info.get("discountPrice") == 0,
        "has_current_promo": len(current_promos) > 0,
        "has_upcoming_promo": len(upcoming_promos) > 0,
    }


def get_all_promotions_info():
    """
    Fetch and extract info for all current Epic Games promotions.

    Returns:
        list: List of cleaned game info dictionaries.
    """
    games = get_epic_promotions()
    return [extract_game_info(game) for game in games]


def save_raw_promotions(games, filepath="data/raw/epic_promotions.json"):
    """
    Save raw Epic Games promotion data to a JSON file.

    Args:
        games (list): Raw list of game dictionaries from Epic API.
        filepath (str): Destination file path for the JSON.

    Returns:
        None
    """
    if not games:
        print("No promotions to save.")
        return
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(games, f, indent=2)
    print(f"Raw promotions saved to {filepath}")