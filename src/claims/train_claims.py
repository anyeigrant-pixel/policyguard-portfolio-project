"""Train a transparent fraud-risk model from the user-provided claims dataset."""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import ARTIFACT_DIR, DATA_DIR

CLAIMS_PATH = DATA_DIR / 'claims' / 'insurance_claims_dataset.csv'
NUMERIC_FEATURES = ['Age', 'Premium_Amount', 'Coverage_Amount', 'Claim_Amount', 'Credit_Score']
CATEGORICAL_FEATURES = ['Gender', 'Region', 'Policy_Type', 'Claim_Type']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train_claims_model():
    """Fit and persist a held-out fraud classifier without the provided risk score."""
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f'Claims data is unavailable at {CLAIMS_PATH}. Add the user-provided claims CSV before training.'
        )
    ARTIFACT_DIR.mkdir(exist_ok=True)
    claims = pd.read_csv(CLAIMS_PATH)
    target = claims['Fraud_Flag'].eq('Yes').astype(int)
    train, test, y_train, y_test = train_test_split(
        claims[FEATURES], target, test_size=.25, stratify=target, random_state=42
    )
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]),
         NUMERIC_FEATURES),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ('preprocess', preprocess),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced')),
    ])
    model.fit(train, y_train)
    probabilities = model.predict_proba(test)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    metrics = {
        'roc_auc': roc_auc_score(y_test, probabilities),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'records': int(len(claims)),
        'positive_rate': float(target.mean()),
    }
    joblib.dump(model, ARTIFACT_DIR / 'claims_fraud_model.joblib')
    (ARTIFACT_DIR / 'claims_metrics.json').write_text(json.dumps(metrics, indent=2))

    feature_names = model.named_steps['preprocess'].get_feature_names_out()
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.named_steps['classifier'].coef_[0],
    })
    importance['absolute_coefficient'] = importance['coefficient'].abs()
    importance.sort_values('absolute_coefficient', ascending=False).head(30).to_csv(
        ARTIFACT_DIR / 'claims_feature_importance.csv', index=False
    )
    return metrics


if __name__ == '__main__':
    print(json.dumps(train_claims_model(), indent=2))
