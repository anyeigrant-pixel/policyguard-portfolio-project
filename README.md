# PolicyGuard

PolicyGuard is a portfolio-ready insurance policy-lapse and retention intelligence platform. **Every policyholder record, service note, label, and metric is synthetic.** It demonstrates an analytics workflow; it is not suitable for underwriting, pricing, claims decisions, or contact prioritization without real-world validation, governance, and fairness review.

## Business problem
Insurers need to understand which policyholders are at elevated risk of non-renewal, what factors contribute to that risk, and whether unstructured service interactions add predictive value beyond structured policy data.

## What it includes
- Reproducible 10,000-row synthetic policyholder dataset
- Held-out structured-only, text-only TF-IDF, and combined structured + NLP lapse models
- Coefficient-based explainability for structured and combined logistic models, plus predictive service-note words/phrases
- Kaplan-Meier retention curve and event-only random-forest time-to-lapse estimate
- Service-note NLP analytics and NMF topic modeling
- K-means customer segmentation with behavioral segment profiles
- Streamlit workflows for customer risk, model comparison, retention levers, segments, NLP, survival, and feature/text drift
- Explicit dashboard errors when data or model artifacts are unavailable
- Automated preprocessing, modeling, inference, and dashboard-service tests

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.generate_data
python -m src.models.train_models
python -m src.nlp.text_analytics
pytest -q
streamlit run app/streamlit_app.py
```

For a headless health check:
```bash
streamlit run app/streamlit_app.py --server.headless true --server.port 8501
curl -fsS http://127.0.0.1:8501/_stcore/health
```

## Modeling and interpretation
All classifiers use a deterministic stratified 75/25 synthetic train/test split. The dashboard reports ROC-AUC, precision, recall, and F1 at a 0.50 threshold. Logistic-regression coefficients on standardized/transformed inputs are the explainability mechanism; positive coefficients raise modeled lapse odds and negative coefficients lower them. This is an interpretable equivalent where SHAP is not installed, avoiding an optional dependency that is not required for reproducible runtime.

The combined model concatenates imputed/scaled structured features, one-hot categories, and TF-IDF unigrams/bigrams. Its improvement is assessed against structured-only ROC-AUC on the held-out synthetic sample. Time-to-lapse MAE is measured only among observed synthetic lapse events, so it is not a full survival model.

## Operations and monitoring
Training writes versionable artifacts to `artifacts/`: models, validation metrics, feature importances, NLP topics, segment profiles, and a numeric/text training baseline. The Monitoring & Drift page compares current mean structured features and mean note length to this baseline; a standardized mean shift of 0.20 or higher is marked for review. A production implementation would add versioned time windows, prediction/calibration monitoring, data-quality checks, alert routing, protected-class fairness review, and retraining approval controls.

## Assumptions and limitations
- Synthetic labels are generated from specified risk relationships, so results cannot establish causal effects or generalize to an insurer.
- Service notes are templated synthetic text; NLP performance therefore overstates what may occur with natural operational notes.
- Customer segments are descriptive K-means clusters, not prescribed treatments.
- Retention interventions should be experimentally evaluated and reviewed for regulatory, fairness, privacy, and customer-impact risks.

See [Demo guide](docs/DEMO.md), [model documentation](docs/MODELS.md), [architecture](docs/ARCHITECTURE.md), and [limitations](docs/LIMITATIONS.md).
