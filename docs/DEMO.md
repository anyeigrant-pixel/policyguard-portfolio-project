# PolicyGuard Demo

PolicyGuard demonstrates **synthetic-data-only** policy-lapse analytics. Begin on **Executive Overview** to frame the retention problem and show the portfolio’s synthetic composition.

## Public portfolio landing page

`index.html` is the static, responsive GitHub Pages landing site. It communicates the business proposition, validated synthetic results, capabilities, workflow, and responsible-use boundary before directing visitors to the source or the Streamlit dashboard. The Streamlit call-to-action is intentionally labeled **deployment pending** until a live app is provisioned.

Preview the site with `python -m http.server 8000`, then browse to `http://127.0.0.1:8000`. To publish, enable GitHub Pages from the `main` branch at the repository root; see the README for the exact settings and expected Pages URL.

## Interactive dashboard

1. **Customer Risk** — choose a synthetic customer to compare structured, text, and combined lapse probabilities, modeled time-to-lapse, globally important structured drivers, and the service note.
2. **Model Comparison** — contrast held-out ROC-AUC and F1 across structured-only, text-only, and combined models; call out the measured combined-model lift.
3. **Retention Analytics** — explore premium changes, payment behavior, tenure, claims, service contacts, and bundling by observed synthetic lapse.
4. **Customer Segments** — explain four standardized-feature K-means clusters as descriptive retention audiences rather than treatment recommendations.
5. **Text Analytics** — review common phrases, words/phrases predictive of synthetic lapse, and NMF service-note topics.
6. **Survival Analysis** — show the Kaplan-Meier retention curve and the event-only time-to-lapse error.
7. **Monitoring & Drift** — explain baseline-versus-current structured and service-text monitoring, including the review threshold.
8. **Claims & Fraud Intelligence** — examine the user-provided claims dataset separately from the synthetic retention workflow, including fraud prevalence, claim characteristics, and a leakage-controlled baseline classifier.

Key interview point: the project tests whether customer-service text contributes incremental predictive signal beyond structured policy data, while making the synthetic-data limitation prominent.
