"""Data access and model-serving utilities shared by the dashboard and tests."""
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'processed' / 'policyholders.csv'
CLAIMS_PATH = ROOT / 'data' / 'claims' / 'insurance_claims_dataset.csv'
ARTIFACT_DIR = ROOT / 'artifacts'
FEATURES = ['age', 'tenure_years', 'annual_premium', 'premium_change_pct',
            'late_payments_12m', 'claims_3y', 'service_tickets_6m',
            'bundled_policies', 'autopay', 'paperless', 'policy_type', 'region']


class ArtifactUnavailableError(RuntimeError):
    """Raised when a dashboard action needs an absent or unreadable artifact."""


def ensure_dashboard_artifacts():
    """Create synthetic dashboard inputs locally when a fresh deployment has none."""
    retention_required = [
        DATA_PATH,
        ARTIFACT_DIR / 'metrics.json',
        ARTIFACT_DIR / 'top_terms.csv',
        ARTIFACT_DIR / 'topics.csv',
    ]
    if not all(path.exists() for path in retention_required):
        from src.data.generate_data import main as generate_data
        from src.models.train_models import main as train_models
        from src.nlp.text_analytics import main as analyze_text

        generate_data()
        train_models()
        analyze_text()
    if CLAIMS_PATH.exists() and not (ARTIFACT_DIR / 'claims_metrics.json').exists():
        from src.claims.train_claims import train_claims_model
        train_claims_model()


def require_artifact(name):
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise ArtifactUnavailableError(
            f"Required artifact '{name}' is unavailable. Run the complete training pipeline first."
        )
    return path


def load_data():
    if not DATA_PATH.exists():
        raise ArtifactUnavailableError(
            'Synthetic policyholder data is unavailable. Run data generation before opening the dashboard.'
        )
    return pd.read_csv(DATA_PATH)


def load_json(name):
    try:
        return json.loads(require_artifact(name).read_text())
    except json.JSONDecodeError as error:
        raise ArtifactUnavailableError(f"Artifact '{name}' is not valid JSON; rerun training.") from error


def load_csv(name):
    return pd.read_csv(require_artifact(name))


def predict_customer(row):
    try:
        structured = joblib.load(require_artifact('structured_churn_model.joblib'))
        text_vectorizer, text_model = joblib.load(require_artifact('text_churn_model.joblib'))
        combo_prep, combo_vectorizer, combo_model = joblib.load(require_artifact('combined_churn_model.joblib'))
        time_columns, time_model = joblib.load(require_artifact('time_to_event_model.joblib'))
    except Exception as error:
        if isinstance(error, ArtifactUnavailableError):
            raise
        raise ArtifactUnavailableError(f"Unable to load model artifact: {error}") from error
    structured_probability = float(structured.predict_proba(row[FEATURES])[:, 1][0])
    text_probability = float(text_model.predict_proba(text_vectorizer.transform(row['service_note']))[:, 1][0])
    combined_matrix = hstack([
        csr_matrix(combo_prep.transform(row[FEATURES])),
        combo_vectorizer.transform(row['service_note']),
    ])
    combined_probability = float(combo_model.predict_proba(combined_matrix)[:, 1][0])
    time_input = pd.get_dummies(row[FEATURES], columns=['policy_type', 'region']).reindex(
        columns=time_columns, fill_value=0
    )
    months = float(time_model.predict(time_input)[0])
    return {
        'structured_probability': structured_probability, 'text_probability': text_probability,
        'combined_probability': combined_probability, 'estimated_months_to_lapse': months,
    }


def top_risk_drivers(row, limit=4):
    importance = load_csv('structured_feature_importance.csv').copy()
    labels = importance['feature'].str.replace('num__', '', regex=False).str.replace('cat__', '', regex=False)
    importance['driver'] = labels
    return importance.nlargest(limit, 'absolute_coefficient')[['driver', 'coefficient']]


def drift_report(current):
    baseline = load_json('monitoring_baseline.json')
    rows = []
    for feature, reference in baseline['numeric'].items():
        current_mean = float(current[feature].mean())
        std = reference['std'] or 1.0
        standardized_shift = abs(current_mean - reference['mean']) / std
        rows.append({'feature': feature, 'baseline_mean': reference['mean'],
                     'current_mean': current_mean, 'standardized_shift': standardized_shift,
                     'status': 'Review' if standardized_shift >= .2 else 'Stable'})
    mean_note_length = float(current['service_note'].str.len().mean())
    text_shift = abs(mean_note_length - baseline['text']['mean_note_length']) / max(
        baseline['text']['mean_note_length'], 1
    )
    rows.append({'feature': 'service_note_length', 'baseline_mean': baseline['text']['mean_note_length'],
                 'current_mean': mean_note_length, 'standardized_shift': text_shift,
                 'status': 'Review' if text_shift >= .2 else 'Stable'})
    return pd.DataFrame(rows)


def load_claims_data():
    if not CLAIMS_PATH.exists():
        raise ArtifactUnavailableError(
            'User-provided claims data is unavailable. Add data/claims/insurance_claims_dataset.csv first.'
        )
    return pd.read_csv(CLAIMS_PATH)


def claims_summary():
    claims = load_claims_data()
    metrics = load_json('claims_metrics.json')
    return claims, metrics
