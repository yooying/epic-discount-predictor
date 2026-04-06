"""
Module for cleaning and transforming raw game data.
Integrates regular expressions for text normalization,
price cleaning, and feature engineering for the
sale likelihood predictor.
"""

import re
import pandas as pd
import numpy as np


def clean_price(price_str):
    """
    Extract a numeric price value from a price string using regex.

    Args:
        price_str (str): Raw price string e.g. '$19.99' or '19.99 USD'.

    Returns:
        float or None: Numeric price value, or None if not extractable.
    """
    if price_str is None:
        return None
    match = re.search(r"\d+\.\d{2}|\d+", str(price_str))
    if match:
        return float(match.group())
    return None


def normalize_title(title):
    """
    Normalize a game title by removing special characters and
    extra whitespace using regex.

    Args:
        title (str): Raw game title string.

    Returns:
        str: Cleaned, lowercased title with only alphanumeric
        characters and spaces.
    """
    if not title:
        return ""
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip().lower()


def extract_discount_percent(text):
    """
    Extract a discount percentage from a raw text string using regex.

    Args:
        text (str): Text containing a discount e.g. 'Save 75% today!'.

    Returns:
        float or None: Discount percentage as a float, or None if
        not found.
    """
    if not text:
        return None
    match = re.search(r"(\d{1,3})\s*%", str(text))
    if match:
        return float(match.group(1))
    return None


def clean_deals_dataframe(df):
    """
    Clean and engineer features on a CheapShark deals DataFrame.

    Args:
        df (pandas.DataFrame): Raw deals DataFrame from CheapShark API.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with engineered features
        including discount flag, days since last change, and game age.
    """
    if df.empty:
        return df

    df = df.copy()

    df["title_clean"] = df["title"].apply(normalize_title)

    df["savings"] = pd.to_numeric(df["savings"], errors="coerce")
    df["normalPrice"] = pd.to_numeric(df["normalPrice"], errors="coerce")
    df["salePrice"] = pd.to_numeric(df["salePrice"], errors="coerce")
    df["metacriticScore"] = pd.to_numeric(
        df["metacriticScore"], errors="coerce"
    )

    df["is_on_sale"] = df["isOnSale"].apply(
        lambda x: 1 if str(x) == "1" else 0
    )

    now = pd.Timestamp.now()
    if "lastChange" in df.columns:
        df["lastChange"] = pd.to_datetime(
            df["lastChange"], unit="s", errors="coerce"
        )
        df["days_since_last_change"] = (
            now - df["lastChange"]
        ).dt.days

    if "releaseDate" in df.columns:
        df["releaseDate"] = pd.to_datetime(
            df["releaseDate"], unit="s", errors="coerce"
        )
        df["game_age_days"] = (now - df["releaseDate"]).dt.days

    df["discount_depth"] = np.where(
        df["normalPrice"] > 0,
        (df["normalPrice"] - df["salePrice"]) / df["normalPrice"] * 100,
        0,
    )

    return df


def clean_epic_dataframe(records):
    """
    Convert and clean a list of Epic Games promotion dictionaries
    into a DataFrame.

    Args:
        records (list): List of extracted Epic Games game info dicts.

    Returns:
        pandas.DataFrame: Cleaned DataFrame of Epic Games promotions.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["title_clean"] = df["title"].apply(normalize_title)
    df["original_price"] = pd.to_numeric(
        df["original_price"], errors="coerce"
    ) / 100
    df["discount_price"] = pd.to_numeric(
        df["discount_price"], errors="coerce"
    ) / 100
    df["effective_date"] = pd.to_datetime(
        df["effective_date"], errors="coerce"
    )
    # Fill NaN values to prevent database insertion errors
    df["discount_amount"] = pd.to_numeric(df["discount_amount"], errors="coerce").fillna(0) / 100
    df["original_price"] = df["original_price"].fillna(0)
    df["discount_price"] = df["discount_price"].fillna(0)
    return df