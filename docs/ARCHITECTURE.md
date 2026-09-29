# Architecture

```mermaid
flowchart LR
  G[Synthetic data generator] --> D[data/processed/policyholders.csv]
  D --> T[Model training]
  D --> N[NLP analytics]
  T --> A[artifacts: models, metrics, importances, segments, monitoring baseline]
  N --> A
  D --> S[Dashboard service]
  A --> S
  S --> U[Streamlit dashboard]
```

`src.services.dashboard_service` is the dashboard boundary: it validates artifacts, loads versioned outputs, produces inference, and calculates baseline drift. Its explicit `ArtifactUnavailableError` prevents success-shaped dashboard output when a pipeline dependency is missing.
