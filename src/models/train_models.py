import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from src.utils.config import DATA_DIR, ARTIFACT_DIR

NUM = ['age','tenure_years','annual_premium','premium_change_pct','late_payments_12m','claims_3y','service_tickets_6m','bundled_policies','autopay','paperless']
CAT = ['policy_type','region']
FEATURES = NUM + CAT

def km_curve(durations, events):
    order = np.argsort(durations)
    t = np.asarray(durations)[order]
    e = np.asarray(events)[order]
    unique_times = np.unique(t[e == 1])
    surv = 1.0
    rows = []
    for time in unique_times:
        at_risk = np.sum(t >= time)
        d = np.sum((t == time) & (e == 1))
        if at_risk > 0:
            surv *= (1 - d / at_risk)
            rows.append({'time_months': float(time), 'survival_probability': float(surv), 'events': int(d), 'at_risk': int(at_risk)})
    return pd.DataFrame(rows)

def classification_metrics(y_true, probability):
    prediction = (probability >= 0.5).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, probability),
        'precision': precision_score(y_true, prediction, zero_division=0),
        'recall': recall_score(y_true, prediction, zero_division=0),
        'f1': f1_score(y_true, prediction, zero_division=0),
    }

def coefficient_importance(feature_names, coefficients, limit=30):
    result = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
    result['absolute_coefficient'] = result['coefficient'].abs()
    return result.sort_values('absolute_coefficient', ascending=False).head(limit)

def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR/'processed'/'policyholders.csv')
    train, test = train_test_split(df, test_size=.25, stratify=df['churned'], random_state=42)
    prep = ColumnTransformer([
        ('num', Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]), NUM),
        ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    pipe = Pipeline([('prep', prep), ('model', model)])
    pipe.fit(train[FEATURES], train['churned'])
    p = pipe.predict_proba(test[FEATURES])[:,1]
    structured = classification_metrics(test['churned'], p)
    joblib.dump(pipe, ARTIFACT_DIR/'structured_churn_model.joblib')
    structured_features = pipe.named_steps['prep'].get_feature_names_out()
    coefficient_importance(
        structured_features, pipe.named_steps['model'].coef_[0]
    ).to_csv(ARTIFACT_DIR/'structured_feature_importance.csv', index=False)

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=5000, stop_words='english')
    Xtr_txt = tfidf.fit_transform(train['service_note'])
    Xte_txt = tfidf.transform(test['service_note'])
    txt_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    txt_model.fit(Xtr_txt, train['churned'])
    pt = txt_model.predict_proba(Xte_txt)[:,1]
    text_metrics = classification_metrics(test['churned'], pt)
    joblib.dump((tfidf, txt_model), ARTIFACT_DIR/'text_churn_model.joblib')
    text_feature_names = tfidf.get_feature_names_out()
    text_importance = coefficient_importance(text_feature_names, txt_model.coef_[0], limit=100)
    text_importance.to_csv(ARTIFACT_DIR/'text_feature_importance.csv', index=False)

    prep_combo = ColumnTransformer([
        ('num', Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]), NUM),
        ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    Xtr_struct = prep_combo.fit_transform(train[FEATURES])
    Xte_struct = prep_combo.transform(test[FEATURES])
    combo_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    combo_model.fit(hstack([csr_matrix(Xtr_struct), Xtr_txt]), train['churned'])
    pc = combo_model.predict_proba(hstack([csr_matrix(Xte_struct), Xte_txt]))[:,1]
    combo = classification_metrics(test['churned'], pc)
    joblib.dump((prep_combo, tfidf, combo_model), ARTIFACT_DIR/'combined_churn_model.joblib')
    combined_features = list(prep_combo.get_feature_names_out()) + list(text_feature_names)
    coefficient_importance(
        combined_features, combo_model.coef_[0]
    ).to_csv(ARTIFACT_DIR/'combined_feature_importance.csv', index=False)

    km_curve(df['months_to_event'], df['event_observed']).to_csv(ARTIFACT_DIR/'kaplan_meier.csv', index=False)
    event_df = df[df['event_observed']==1].copy()
    Xs = pd.get_dummies(event_df[FEATURES], columns=CAT, drop_first=False)
    surv_model = RandomForestRegressor(n_estimators=250, min_samples_leaf=8, random_state=42, n_jobs=-1)
    Xtr, Xte, ytr, yte = train_test_split(Xs, event_df['months_to_event'], test_size=.25, random_state=42)
    surv_model.fit(Xtr, ytr)
    pred_time = surv_model.predict(Xte)
    time_mae = mean_absolute_error(yte, pred_time)
    joblib.dump((list(Xs.columns), surv_model), ARTIFACT_DIR/'time_to_event_model.joblib')

    segment_scaler = StandardScaler()
    segment_matrix = segment_scaler.fit_transform(df[NUM])
    segment_model = KMeans(n_clusters=4, random_state=42, n_init=20)
    df['segment'] = segment_model.fit_predict(segment_matrix)
    segment_names = {
        0: 'Established value seekers', 1: 'High-touch risk', 2: 'Stable digital loyalists',
        3: 'Premium-sensitive newcomers',
    }
    df['segment_name'] = df['segment'].map(segment_names)
    df[['customer_id', 'segment', 'segment_name']].to_csv(ARTIFACT_DIR/'customer_segments.csv', index=False)
    segment_profile = df.groupby(['segment', 'segment_name'], as_index=False).agg(
        customers=('customer_id', 'size'), lapse_rate=('churned', 'mean'),
        avg_tenure_years=('tenure_years', 'mean'), avg_premium_change_pct=('premium_change_pct', 'mean'),
        avg_late_payments=('late_payments_12m', 'mean'), avg_service_tickets=('service_tickets_6m', 'mean'),
        bundled_rate=('bundled_policies', 'mean'), autopay_rate=('autopay', 'mean'),
    )
    segment_profile.to_csv(ARTIFACT_DIR/'segment_profiles.csv', index=False)
    joblib.dump((segment_scaler, segment_model, segment_names), ARTIFACT_DIR/'segment_model.joblib')

    baseline = {
        'numeric': {column: {'mean': float(df[column].mean()), 'std': float(df[column].std())} for column in NUM},
        'text': {'mean_note_length': float(df['service_note'].str.len().mean()),
                 'vocabulary_size': int(len(text_feature_names))},
    }
    (ARTIFACT_DIR/'monitoring_baseline.json').write_text(json.dumps(baseline, indent=2))
    metrics = {
        'structured': structured, 'text_only': text_metrics, 'combined': combo,
        'time_to_event': {'mae_months': time_mae},
        'segmentation': {'clusters': 4, 'silhouette_score': silhouette_score(segment_matrix, df['segment'])},
        'nlp_improves_roc_auc': combo['roc_auc'] > structured['roc_auc'],
    }
    (ARTIFACT_DIR/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
