"""Feature engineering and preprocessing pipelines for Real-Time Content Moderator."""
from __future__ import annotations

import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_FEATURE = 'comment_text'

NUMERICAL_FEATURES = [
    'comment_length',
    'word_count',
    'uppercase_ratio',
    'exclamation_count',
    'question_count',
    'special_character_count',
    'digit_count',
    'unique_word_ratio',
    'average_word_length',
]

CATEGORICAL_FEATURES = ['time_of_day']

SURFACE_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
HYBRID_FEATURES = [TEXT_FEATURE] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
ALL_FEATURES = HYBRID_FEATURES


def extract_text_features(df: pd.DataFrame, text_column: str = 'comment_text') -> pd.DataFrame:
    """Extract 9 surface and syntactic text features from raw comment text."""
    df_out = df.copy()
    text = df_out[text_column].fillna('').astype(str)

    lengths = text.str.len()
    words = text.str.split()
    word_counts = words.str.len().fillna(0).astype(int)

    df_out['comment_length'] = lengths.astype(float)
    df_out['word_count'] = word_counts.astype(float)

    # Uppercase ratio
    uppercase_counts = text.apply(lambda s: sum(1 for c in s if c.isupper()))
    df_out['uppercase_ratio'] = (uppercase_counts / (lengths + 1)).astype(float)

    # Punctuation counts
    df_out['exclamation_count'] = text.str.count(r'!').astype(float)
    df_out['question_count'] = text.str.count(r'\?').astype(float)

    # Special characters and digits
    df_out['special_character_count'] = text.apply(lambda s: sum(1 for c in s if not c.isalnum() and not c.isspace())).astype(float)
    df_out['digit_count'] = text.apply(lambda s: sum(1 for c in s if c.isdigit())).astype(float)

    # Lexical richness and average word length
    df_out['unique_word_ratio'] = words.apply(lambda w: len(set(w)) / (len(w) + 1e-5) if len(w) > 0 else 0.0).astype(float)
    df_out['average_word_length'] = (lengths / (word_counts + 1e-5)).astype(float)

    return df_out


def extract_time_of_day(date_series: pd.Series) -> pd.Series:
    """Categorize timestamp into Morning, Afternoon, Evening, Night, or Unknown."""
    dt = pd.to_datetime(date_series, errors='coerce')
    hours = dt.dt.hour

    def _cat(h):
        if pd.isna(h):
            return 'Unknown'
        elif 5 <= h < 12:
            return 'Morning'
        elif 12 <= h < 17:
            return 'Afternoon'
        elif 17 <= h < 21:
            return 'Evening'
        else:
            return 'Night'

    return hours.apply(_cat)


def get_current_time_category() -> str:
    """Return time-of-day category for current wall clock."""
    h = datetime.now().hour
    if 5 <= h < 12:
        return 'Morning'
    elif 12 <= h < 17:
        return 'Afternoon'
    elif 17 <= h < 21:
        return 'Evening'
    return 'Night'


def build_preprocessor(
    num_cols: list[str] = NUMERICAL_FEATURES,
    cat_cols: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build a leakage-free ColumnTransformer for numerical scaling and categorical encoding (Surface baseline)."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols),
        ]
    )


def build_hybrid_preprocessor(
    text_col: str = TEXT_FEATURE,
    num_cols: list[str] = NUMERICAL_FEATURES,
    cat_cols: list[str] = CATEGORICAL_FEATURES,
    max_text_features: int = 30000,
) -> ColumnTransformer:
    """Build a comprehensive ColumnTransformer combining TF-IDF word/char n-grams with scaled surface features."""
    text_transformer = TfidfVectorizer(
        max_features=max_text_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode'
    )

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('text', text_transformer, text_col),
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols),
        ]
    )


def extract_single_comment_features(
    text: str,
    time_of_day: str | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Extract full feature row for a single comment string for instant inference in dashboard/API."""
    text_val = str(text or '')
    words = text_val.split()
    length = len(text_val)
    w_count = len(words)
    tod = time_of_day or get_current_time_category()

    row = {
        'comment_text': text_val,
        'comment_length': float(length),
        'word_count': float(w_count),
        'uppercase_ratio': float(sum(1 for c in text_val if c.isupper()) / max(length + 1, 1)),
        'exclamation_count': float(text_val.count('!')),
        'question_count': float(text_val.count('?')),
        'special_character_count': float(len(re.findall(r'[^A-Za-z0-9\s]', text_val))),
        'digit_count': float(sum(1 for c in text_val if c.isdigit())),
        'unique_word_ratio': float(len(set(w.lower() for w in words)) / max(w_count, 1)) if w_count > 0 else 0.0,
        'average_word_length': float(length / max(w_count, 1)) if w_count > 0 else 0.0,
        'time_of_day': str(tod),
    }

    cols = feature_columns or HYBRID_FEATURES
    # Filter to only the requested feature columns if specified
    filtered_row = {k: row[k] for k in cols if k in row}
    return pd.DataFrame([filtered_row], columns=cols)
