"""
Module for managing the SQLite database using SQLAlchemy.
Handles creation, insertion, and retrieval of game deals,
Epic Games promotions, and Metacritic review data.
"""

import pandas as pd
import os
from sqlalchemy import (
    create_engine,
    text,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, Session


Base = declarative_base()

DATABASE_URL = "sqlite:///" + os.path.join(
    os.getcwd(), "epic-discount-predictor", "database", "games.db"
).replace("\\", "/")


class Deal(Base):
    """ORM model for CheapShark deals table."""

    __tablename__ = "deals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, unique=True)
    title = Column(String)
    title_clean = Column(String)
    sale_price = Column(Float)
    normal_price = Column(Float)
    savings = Column(Float)
    is_on_sale = Column(Integer)
    metacritic_score = Column(Float)
    metacritic_link = Column(String)
    release_date = Column(DateTime)
    last_change = Column(DateTime)
    days_since_last_change = Column(Integer)
    game_age_days = Column(Integer)
    discount_depth = Column(Float)
    deal_rating = Column(Float)


class GameDetail(Base):
    """ORM model for Epic Games promotions table."""

    __tablename__ = "game_details"

    id = Column(Integer, primary_key=True, autoincrement=True)
    epic_id = Column(String, unique=True)
    title = Column(String)
    title_clean = Column(String)
    description = Column(Text)
    status = Column(String)
    effective_date = Column(DateTime)
    original_price = Column(Float)
    discount_price = Column(Float)
    discount_amount = Column(Float)
    currency = Column(String)
    seller = Column(String)
    is_free = Column(Boolean)
    has_current_promo = Column(Boolean)
    has_upcoming_promo = Column(Boolean)


class Review(Base):
    """ORM model for Metacritic reviews table."""

    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metacritic_link = Column(String)
    title = Column(String)
    metascore = Column(Integer)
    summary = Column(Text)
    raw_review_blob = Column(Text)


def get_engine():
    """
    Create and return a SQLAlchemy engine connected to the SQLite database.

    Returns:
        sqlalchemy.engine.Engine: Connected database engine.
    """
    engine = create_engine(DATABASE_URL, echo=False)
    return engine


def init_db():
    """
    Initialize the database by creating all tables if they
    do not already exist.

    Returns:
        sqlalchemy.engine.Engine: The engine connected to the database.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database initialized successfully.")
    return engine


def insert_deals(df, engine):
    """
    Insert cleaned deals DataFrame into the deals table.
    Skips rows with duplicate game IDs.

    Args:
        df (pandas.DataFrame): Cleaned deals DataFrame.
        engine (sqlalchemy.engine.Engine): Connected database engine.

    Returns:
        None
    """
    if df.empty:
        print("No deals to insert.")
        return

    column_map = {
        "gameID": "game_id",
        "title": "title",
        "title_clean": "title_clean",
        "salePrice": "sale_price",
        "normalPrice": "normal_price",
        "savings": "savings",
        "is_on_sale": "is_on_sale",
        "metacriticScore": "metacritic_score",
        "metacriticLink": "metacritic_link",
        "releaseDate": "release_date",
        "lastChange": "last_change",
        "days_since_last_change": "days_since_last_change",
        "game_age_days": "game_age_days",
        "discount_depth": "discount_depth",
        "dealRating": "deal_rating",
    }

    df_renamed = df.rename(columns=column_map)
    valid_cols = [c for c in column_map.values() if c in df_renamed.columns]
    df_to_insert = df_renamed[valid_cols]

    with Session(engine) as session:
        for _, row in df_to_insert.iterrows():
            exists = session.query(Deal).filter_by(
                game_id=row.get("game_id")
            ).first()
            if not exists:
                session.add(Deal(**row.to_dict()))
        session.commit()
    print(f"Inserted deals into database.")


def insert_game_details(df, engine):
    """
    Insert cleaned Epic Games promotions DataFrame into game_details table.
    Skips rows with duplicate epic IDs.

    Args:
        df (pandas.DataFrame): Cleaned Epic promotions DataFrame.
        engine (sqlalchemy.engine.Engine): Connected database engine.

    Returns:
        None
    """
    if df.empty:
        print("No game details to insert.")
        return

    column_map = {
        "id": "epic_id",
        "title": "title",
        "title_clean": "title_clean",
        "description": "description",
        "status": "status",
        "effective_date": "effective_date",
        "expiry_date": "expiry_date",
        "original_price": "original_price",
        "discount_price": "discount_price",
        "discount_amount": "discount_amount",
        "currency": "currency",
        "seller": "seller",
        "is_free": "is_free",
        "has_current_promo": "has_current_promo",
        "has_upcoming_promo": "has_upcoming_promo",
    }

    df_renamed = df.rename(columns=column_map)
    valid_cols = [c for c in column_map.values() if c in df_renamed.columns]
    df_to_insert = df_renamed[valid_cols]

    with Session(engine) as session:
        for _, row in df_to_insert.iterrows():
            exists = session.query(GameDetail).filter_by(
                epic_id=str(row.get("epic_id"))
            ).first()
            if not exists:
                row_dict = row.to_dict()
                for key, value in row_dict.items():
                    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
                        row_dict[key] = None
                session.add(GameDetail(**row_dict))
        session.commit()
    print(f"Inserted game details into database.")


def insert_reviews(reviews, engine):
    """
    Insert a list of scraped Metacritic review dictionaries into
    the reviews table.

    Args:
        reviews (list): List of review dictionaries from scraper module.
        engine (sqlalchemy.engine.Engine): Connected database engine.

    Returns:
        None
    """
    if not reviews:
        print("No reviews to insert.")
        return

    with Session(engine) as session:
        for review in reviews:
            exists = session.query(Review).filter_by(
                metacritic_link=review.get("metacritic_link")
            ).first()
            if not exists:
                session.add(Review(
                    metacritic_link=review.get("metacritic_link"),
                    title=review.get("title"),
                    metascore=review.get("metascore"),
                    summary=review.get("summary"),
                    raw_review_blob=review.get("raw_review_blob"),
                ))
        session.commit()
    print(f"Inserted reviews into database.")


def query_deals_with_reviews(engine):
    """
    Query and join deals and reviews tables on metacritic_link.

    Args:
        engine (sqlalchemy.engine.Engine): Connected database engine.

    Returns:
        pandas.DataFrame: Joined DataFrame of deals and their
        corresponding Metacritic reviews.
    """
    query = text("""
        SELECT
            d.title,
            d.sale_price,
            d.normal_price,
            d.savings,
            d.is_on_sale,
            d.metacritic_score,
            d.game_age_days,
            d.days_since_last_change,
            d.discount_depth,
            r.metascore,
            r.summary,
            r.raw_review_blob
        FROM deals d
        LEFT JOIN reviews r
            ON d.metacritic_link = r.metacritic_link
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def query_all_deals(engine):
    """
    Retrieve all records from the deals table.

    Args:
        engine (sqlalchemy.engine.Engine): Connected database engine.

    Returns:
        pandas.DataFrame: All deals from the database.
    """
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM deals"), conn)
    return df


def query_all_game_details(engine):
    """
    Retrieve all records from the game_details table.

    Args:
        engine (sqlalchemy.engine.Engine): Connected database engine.

    Returns:
        pandas.DataFrame: All Epic Games promotions from the database.
    """
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM game_details"), conn)
    return df