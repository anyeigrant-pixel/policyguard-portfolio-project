import pandas as pd
from pathlib import Path
import pytest
from src.data.generate_data import make_note
from src.models.train_models import classification_metrics, km_curve
from src.services.dashboard_service import (
    ArtifactUnavailableError, drift_report, load_data, load_json, predict_customer,
)

def test_dataset_exists():
    p = Path('data/processed/policyholders.csv')
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) >= 5000
    assert {'service_note','churned','months_to_event'}.issubset(df.columns)

def test_metrics_exist():
    p = Path('artifacts/metrics.json')
    assert p.exists()

def test_generated_note_contains_relevant_synthetic_signal():
    row = pd.Series({
        'premium_change_pct': 20, 'late_payments_12m': 2, 'service_tickets_6m': 0,
        'claims_3y': 0, 'bundled_policies': 1, 'tenure_years': 3, 'churned': 1,
    })
    note = make_note(row)
    assert 'premium increase' in note
    assert 'payment timing' in note

def test_metrics_and_kaplan_meier_shape():
    metrics = classification_metrics(pd.Series([0, 1, 0, 1]), pd.Series([.1, .9, .3, .8]))
    assert metrics['roc_auc'] == 1.0
    curve = km_curve([1, 2, 3], [1, 0, 1])
    assert list(curve.columns) == ['time_months', 'survival_probability', 'events', 'at_risk']
    assert curve.survival_probability.is_monotonic_decreasing

def test_inference_and_monitoring_return_expected_values():
    df = load_data()
    prediction = predict_customer(df.iloc[[0]])
    assert set(prediction) == {
        'structured_probability', 'text_probability', 'combined_probability',
        'estimated_months_to_lapse',
    }
    assert all(0 <= prediction[key] <= 1 for key in prediction if key.endswith('probability'))
    report = drift_report(df.head(200))
    assert {'feature', 'standardized_shift', 'status'}.issubset(report.columns)
    assert 'service_note_length' in report.feature.values

def test_nlp_importance_artifact_has_predictive_phrases():
    importance = pd.read_csv('artifacts/text_feature_importance.csv')
    assert {'feature', 'coefficient', 'absolute_coefficient'}.issubset(importance.columns)
    assert (importance.absolute_coefficient > 0).all()

def test_dashboard_fails_explicitly_for_missing_artifact(monkeypatch, tmp_path):
    import src.services.dashboard_service as service
    monkeypatch.setattr(service, 'ARTIFACT_DIR', tmp_path)
    with pytest.raises(ArtifactUnavailableError, match='complete training pipeline'):
        load_json('metrics.json')

def test_inference_fails_explicitly_when_model_is_missing(monkeypatch, tmp_path):
    import src.services.dashboard_service as service
    row = load_data().iloc[[0]]
    monkeypatch.setattr(service, 'ARTIFACT_DIR', tmp_path)
    with pytest.raises(ArtifactUnavailableError, match='complete training pipeline'):
        predict_customer(row)
