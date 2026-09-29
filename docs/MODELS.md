# Model documentation

## Classification

The pipeline creates a fixed random 75/25 stratified split of the synthetic data (`random_state=42`).

| Model | Inputs | Estimator |
| --- | --- | --- |
| Structured-only | Numeric policy/payment/service features plus policy type and region | Imputed, scaled/one-hot logistic regression |
| Text-only | Synthetic service-note TF-IDF unigrams and bigrams | Logistic regression |
| Combined | Structured transformations concatenated with TF-IDF | Logistic regression |

Metrics are held-out ROC-AUC, precision, recall, and F1 using a 0.50 probability threshold. Coefficients are saved as feature-importance artifacts. Positive coefficient direction indicates greater modeled lapse odds; coefficient magnitudes are not causal effects or customer-specific explanations.

## Time and segmentation

The retention curve is a Kaplan-Meier-style calculation over synthetic durations. A random forest estimates months-to-event among observed events and reports MAE. This is not a Cox model and does not provide a production-grade censored survival prediction.

Four K-means clusters are fitted after standardizing numeric structured features. Segment labels summarize the synthetic profile and must not be treated as customer actions or eligibility decisions.
