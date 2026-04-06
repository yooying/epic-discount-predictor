"""
Module for fetching game deal data from the CheapShark API.
CheapShark tracks game prices and discounts across multiple stores.
Store ID 25 = Epic Games Store.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("CHEAPSHARK_BASE_URL")
EPIC_STORE_ID = "25"


def get_epic_deals(page_size=60, page_number=0):
    """
    Fetch current deals from the Epic Games Store via CheapShark API.

    Args:
        page_size (int): Number of deals to fetch per page. Max 60.
        page_number (int): Page number for pagination.

    Returns:
        list: A list of deal dictionaries, or empty list if request fails.
    """
    params = {
        "storeID": EPIC_STORE_ID,
        "pageSize": page_size,
        "pageNumber": page_number,
    }
    try:
        response = requests.get(f"{BASE_URL}/deals", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching deals: {e}")
        return []


def get_all_epic_deals(max_pages=5):
    """
    Fetch multiple pages of Epic Games deals from CheapShark API.

    Args:
        max_pages (int): Maximum number of pages to fetch.

    Returns:
        list: A combined list of all deal dictionaries across pages.
    """
    all_deals = []
    for page in range(max_pages):
        print(f"Fetching page {page + 1} of {max_pages}...")
        deals = get_epic_deals(page_size=60, page_number=page)
        if not deals:
            break
        all_deals.extend(deals)
        time.sleep(1)
    return all_deals


def deals_to_dataframe(deals):
    """
    Convert a list of deal dictionaries into a pandas DataFrame.

    Args:
        deals (list): List of deal dictionaries from CheapShark API.

    Returns:
        pandas.DataFrame: DataFrame containing deal data, or empty
        DataFrame if input is empty.
    """
    if not deals:
        print("No deals to convert.")
        return pd.DataFrame()
    df = pd.DataFrame(deals)
    df["salePrice"] = pd.to_numeric(df["salePrice"], errors="coerce")
    df["normalPrice"] = pd.to_numeric(df["normalPrice"], errors="coerce")
    df["savings"] = pd.to_numeric(df["savings"], errors="coerce")
    df["metacriticScore"] = pd.to_numeric(df["metacriticScore"], errors="coerce")
    df["releaseDate"] = pd.to_datetime(df["releaseDate"], unit="s", errors="coerce")
    df["lastChange"] = pd.to_datetime(df["lastChange"], unit="s", errors="coerce")
    return df


def save_deals_to_csv(df, filepath="data/processed/epic_deals.csv"):
    """
    Save a deals DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): DataFrame of deals to save.
        filepath (str): Destination file path for the CSV.

    Returns:
        None
    """
    if df.empty:
        print("DataFrame is empty. Nothing saved.")
        return
    df.to_csv(filepath, index=False)
    print(f"Deals saved to {filepath}")


def save_deals_to_json(deals, filepath="data/raw/epic_deals.json"):
    """
    Save raw deal data to a JSON file.

    Args:
        deals (list): Raw list of deal dictionaries.
        filepath (str): Destination file path for the JSON.

    Returns:
        None
    """
    import json
    if not deals:
        print("No deals to save.")
        return
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(deals, f, indent=2)
    print(f"Raw deals saved to {filepath}")