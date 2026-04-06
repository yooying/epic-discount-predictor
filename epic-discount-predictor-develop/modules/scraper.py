"""
Module for scraping game review data from Metacritic.
Uses BeautifulSoup to extract review scores and review text
from Metacritic game pages.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.getenv("METACRITIC_BASE_URL")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_game_page(metacritic_link):
    """
    Scrape a Metacritic game page and extract review information.

    Args:
        metacritic_link (str): Relative Metacritic link e.g.
        '/game/star-wars-jedi-survivor/'.

    Returns:
        dict: Dictionary containing scraped game data including
        title, scores, and review text. Returns None if scraping fails.
    """
    url = f"{BASE_URL}{metacritic_link}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html5lib")
        return parse_game_page(soup, metacritic_link)
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None


def parse_game_page(soup, metacritic_link):
    """
    Parse a BeautifulSoup object from a Metacritic game page.

    Args:
        soup (BeautifulSoup): Parsed HTML of the Metacritic page.
        metacritic_link (str): The relative link used to identify the game.

    Returns:
        dict: Extracted fields including title, metascore, user score,
        summary, and raw review text.
    """
    title = None
    title_tag = soup.find("h1")
    if title_tag:
        title = title_tag.get_text(strip=True)

    metascore = None
    metascore_tag = soup.find("span", attrs={"data-v-4cdca868": True})
    if metascore_tag:
        raw = metascore_tag.get_text(strip=True)
        match = re.search(r"\d+", raw)
        if match:
            metascore = int(match.group())

    summary = None
    summary_tag = soup.find("span", class_=re.compile(r"description"))
    if summary_tag:
        summary = summary_tag.get_text(strip=True)

    review_texts = []
    review_tags = soup.find_all("div", class_=re.compile(r"review_body|c-siteReview"))
    for tag in review_tags[:10]:
        text = tag.get_text(strip=True)
        if text:
            review_texts.append(text)

    return {
        "metacritic_link": metacritic_link,
        "title": title,
        "metascore": metascore,
        "summary": summary,
        "review_texts": review_texts,
        "raw_review_blob": " ".join(review_texts),
    }


def scrape_multiple_games(metacritic_links, delay=2):
    """
    Scrape multiple Metacritic game pages with a delay between requests.

    Args:
        metacritic_links (list): List of relative Metacritic links.
        delay (int): Seconds to wait between requests. Default is 2.

    Returns:
        list: List of scraped game dictionaries. Failed scrapes are excluded.
    """
    results = []
    for i, link in enumerate(metacritic_links):
        print(f"Scraping {i + 1}/{len(metacritic_links)}: {link}")
        data = scrape_game_page(link)
        if data:
            results.append(data)
        time.sleep(delay)
    return results


#delete
def extract_score_from_text(text):
    """
    Extract a numerical score from a raw text string using regex.

    Args:
        text (str): Raw text that may contain a score like '85/100' or '8.5'.

    Returns:
        float or None: Extracted score as a float, or None if not found.
    """
    if not text:
        return None
    match = re.search(r"(\d{1,3}(?:\.\d)?)\s*/\s*100", text)
    if match:
        return float(match.group(1))
    match = re.search(r"\b(\d\.\d|\d{2,3})\b", text)
    if match:
        return float(match.group(1))
    return None