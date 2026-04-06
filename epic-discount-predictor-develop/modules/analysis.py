"""
Module for exploratory data analysis and sale likelihood prediction.
Uses scikit-learn to build a model that predicts whether a game
is likely to go on sale based on its features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "normal_price",
    "savings",
    "metacritic_score",
    "game_age_days",
    "days_since_last_change",
    "discount_depth",
]
TARGET_COL = "is_on_sale"


def get_feature_matrix(df):
    """
    Extract feature matrix and target vector from a DataFrame.
    Drops rows with missing values in feature or target columns.

    Args:
        df (pandas.DataFrame): Cleaned deals DataFrame with
        required feature columns.

    Returns:
        tuple: (X, y) where X is the feature DataFrame and
        y is the target Series.
    """
    cols_needed = FEATURE_COLS + [TARGET_COL]
    available = [c for c in cols_needed if c in df.columns]
    df_model = df[available].dropna()
    X = df_model[[c for c in FEATURE_COLS if c in df_model.columns]]
    y = df_model[TARGET_COL]
    return X, y


def train_sale_predictor(X, y):
    """
    Train a Random Forest classifier to predict sale likelihood.

    Args:
        X (pandas.DataFrame): Feature matrix.
        y (pandas.Series): Target vector (1 = on sale, 0 = not on sale).

    Returns:
        tuple: (model, X_test, y_test, scaler) trained model,
        test split, and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model, X_test, y_test, scaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model and print classification metrics.

    Args:
        model: Trained scikit-learn classifier.
        X_test (array): Scaled test feature matrix.
        y_test (pandas.Series): True target values for test set.

    Returns:
        dict: Dictionary containing accuracy, roc_auc, and
        classification report string.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = model.score(X_test, y_test)

    # ROC AUC requires two classes — skip if only one class present
    unique_classes = len(set(y_test))
    if unique_classes > 1:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None
        print("Note: ROC AUC not computed — only one class in test set.")

    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(report)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "report": report,
    }


def predict_sale_likelihood(game_features, model, scaler):
    """
    Predict the sale likelihood probability for a single game.

    Args:
        game_features (dict): Dictionary of feature values for one game.
        model: Trained scikit-learn classifier.
        scaler: Fitted StandardScaler used during training.

    Returns:
        float: Probability (0 to 1) that the game will go on sale.
    """
    df = pd.DataFrame([game_features])
    df = df[[c for c in FEATURE_COLS if c in df.columns]]
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    X_scaled = scaler.transform(df)
    prob = model.predict_proba(X_scaled)[0][1]
    return round(prob, 4)


def plot_discount_distribution(df):
    """
    Plot the distribution of discount percentages across games.

    Args:
        df (pandas.DataFrame): Deals DataFrame with savings column.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        df["savings"].dropna(),
        bins=30,
        kde=True,
        color="#6366f1",
        ax=ax,
    )
    ax.set_title("Distribution of Discount Percentages")
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Number of Games")
    plt.tight_layout()
    return fig


def plot_price_vs_discount(df):
    """
    Plot a scatter of original price vs discount percentage.

    Args:
        df (pandas.DataFrame): Deals DataFrame with normalPrice
        and savings columns.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df,
        x="normalPrice",
        y="savings",
        hue="is_on_sale",
        palette={0: "#94a3b8", 1: "#6366f1"},
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Original Price vs Discount Percentage")
    ax.set_xlabel("Original Price (USD)")
    ax.set_ylabel("Discount (%)")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names):
    """
    Plot feature importances from a trained Random Forest model.

    Args:
        model (RandomForestClassifier): Trained Random Forest model.
        feature_names (list): List of feature column names.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=sorted_importances,
        y=sorted_features,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Feature Importances — Sale Likelihood Predictor")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot a confusion matrix for the trained model.

    Args:
        model: Trained scikit-learn classifier.
        X_test (array): Scaled test feature matrix.
        y_test (pandas.Series): True target values.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Not On Sale", "On Sale"],
        yticklabels=["Not On Sale", "On Sale"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig