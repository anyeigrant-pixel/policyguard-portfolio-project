import sys
import importlib.util
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = ROOT / 'src' / 'services' / 'dashboard_service.py'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
service_spec = importlib.util.spec_from_file_location('policyguard_dashboard_service', SERVICE_PATH)
if service_spec is None or service_spec.loader is None:
    raise RuntimeError(f'Unable to load dashboard service from {SERVICE_PATH}.')
dashboard_service = importlib.util.module_from_spec(service_spec)
service_spec.loader.exec_module(dashboard_service)

ArtifactUnavailableError = dashboard_service.ArtifactUnavailableError
claims_summary = dashboard_service.claims_summary
drift_report = dashboard_service.drift_report
ensure_dashboard_artifacts = dashboard_service.ensure_dashboard_artifacts
load_csv = dashboard_service.load_csv
load_data = dashboard_service.load_data
load_json = dashboard_service.load_json
predict_customer = dashboard_service.predict_customer
top_risk_drivers = dashboard_service.top_risk_drivers

PINK_PALETTE = ['#D84A8C', '#A82B68', '#F08AB5', '#6E315D', '#F4B5CF', '#C75B8E']
PLOTLY_THEME = go.layout.Template(pio.templates['plotly_white'])
PLOTLY_THEME.layout.colorway = PINK_PALETTE
PLOTLY_THEME.layout.paper_bgcolor = '#FFF8FC'
PLOTLY_THEME.layout.plot_bgcolor = '#FFF8FC'
PLOTLY_THEME.layout.font = {'color': '#35152A'}
PLOTLY_THEME.layout.xaxis = {'gridcolor': '#F1D7E4', 'linecolor': '#DFAEC8'}
PLOTLY_THEME.layout.yaxis = {'gridcolor': '#F1D7E4', 'linecolor': '#DFAEC8'}
px.defaults.template = PLOTLY_THEME
px.defaults.color_discrete_sequence = PINK_PALETTE

st.set_page_config(page_title='PolicyGuard | Retention Intelligence', page_icon='🛡️', layout='wide')
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(244, 173, 207, 0.25), transparent 32rem),
            #fff8fc;
    }
    .block-container {padding-top: 2rem; max-width: 1400px;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #fff, #fff1f7);
        border: 1px solid #f2c9dc;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(129, 35, 82, 0.08);
    }
    [data-testid="stMetricLabel"] {color: #7a315b;}
    [data-testid="stMetricValue"] {color: #9d285f;}
    [data-testid="stSidebar"] {background: #fff1f7;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color: #4d1937;}
</style>
""", unsafe_allow_html=True)
st.title('PolicyGuard')
st.caption('Insurance lapse intelligence built exclusively with synthetic policyholder data — not for production underwriting or customer decisions.')


def required_data():
    try:
        return load_data()
    except (ArtifactUnavailableError, OSError, pd.errors.ParserError) as error:
        st.error(f'Dashboard data is unavailable: {error}')
        st.stop()


def artifact_page(action):
    try:
        action()
    except ArtifactUnavailableError as error:
        st.error(str(error))
    except (OSError, ValueError) as error:
        st.error(f'Unable to display this analysis. Verify pipeline artifacts and rerun training. Detail: {error}')


with st.spinner('Preparing synthetic policy and model artifacts...'):
    try:
        ensure_dashboard_artifacts()
    except (OSError, ValueError) as error:
        st.error(f'Unable to prepare synthetic dashboard artifacts: {error}')
        st.stop()

df = required_data()
pages = ['Executive Overview', 'Customer Risk', 'Model Comparison', 'Retention Analytics',
         'Customer Segments', 'Text Analytics', 'Survival Analysis', 'Monitoring & Drift',
         'Claims & Fraud Intelligence']
page = st.sidebar.radio('Navigate', pages)

if page == 'Executive Overview':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Synthetic policies', f'{len(df):,}')
    c2.metric('Observed lapse', f'{df.churned.mean():.1%}')
    c3.metric('Average premium', f'${df.annual_premium.mean():,.0f}')
    c4.metric('Autopay adoption', f'{df.autopay.mean():.1%}')
    st.subheader('Where retention risk concentrates')
    st.plotly_chart(px.histogram(df, x='premium_change_pct', color='churned', barmode='overlay',
                                 labels={'churned': 'Lapsed'}, title='Premium change versus observed lapse'),
                    use_container_width=True)
    st.info('Use this portfolio demonstration to explore patterns and model behavior. All records and labels are synthetic.')

elif page == 'Customer Risk':
    def customer_risk():
        customer_id = st.selectbox('Synthetic customer ID', df.customer_id)
        row = df.loc[df.customer_id.eq(customer_id)].iloc[[0]]
        result = predict_customer(row)
        c1, c2, c3 = st.columns(3)
        c1.metric('Combined lapse probability', f"{result['combined_probability']:.1%}")
        c2.metric('Structured-only estimate', f"{result['structured_probability']:.1%}")
        c3.metric('Estimated months to lapse', f"{result['estimated_months_to_lapse']:.1f}")
        st.subheader('Main modeled risk drivers')
        drivers = top_risk_drivers(row)
        drivers['direction'] = drivers.coefficient.map(lambda x: 'Higher lapse risk' if x > 0 else 'Lower lapse risk')
        st.dataframe(drivers[['driver', 'direction']], hide_index=True, use_container_width=True)
        st.subheader('Relevant service interaction')
        st.write(row.service_note.iloc[0])
        st.caption(f"NLP lapse signal: {result['text_probability']:.1%}. The driver list reflects global logistic-model coefficients; it is an interpretable coefficient-based alternative to SHAP.")
    artifact_page(customer_risk)

elif page == 'Model Comparison':
    def model_comparison():
        metrics = load_json('metrics.json')
        comparison = pd.DataFrame([
            {'Model': 'Structured-only', **metrics['structured']},
            {'Model': 'Text-only', **metrics['text_only']},
            {'Model': 'Combined structured + NLP', **metrics['combined']},
        ])
        st.subheader('Validation performance')
        st.dataframe(comparison.style.format({'roc_auc': '{:.3f}', 'precision': '{:.3f}', 'recall': '{:.3f}', 'f1': '{:.3f}'}),
                     hide_index=True, use_container_width=True)
        st.plotly_chart(px.bar(comparison, x='Model', y=['roc_auc', 'f1'], barmode='group',
                               title='ROC-AUC and F1 by modeling approach'), use_container_width=True)
        lift = metrics['combined']['roc_auc'] - metrics['structured']['roc_auc']
        st.success(f"Adding service-note NLP {'improves' if lift > 0 else 'does not improve'} ROC-AUC by {lift:+.3f} on the held-out synthetic test set.")
    artifact_page(model_comparison)

elif page == 'Retention Analytics':
    st.subheader('Retention levers')
    left, right = st.columns(2)
    left.plotly_chart(px.box(df, x='churned', y='premium_change_pct', color='churned',
                             title='Premium change by lapse outcome'), use_container_width=True)
    right.plotly_chart(px.histogram(df, x='late_payments_12m', color='churned', barmode='group',
                                   title='Payment behavior by lapse outcome'), use_container_width=True)
    factors = df.groupby('churned', as_index=False).agg(
        tenure_years=('tenure_years', 'mean'), claims_3y=('claims_3y', 'mean'),
        service_tickets_6m=('service_tickets_6m', 'mean'), bundled_policies=('bundled_policies', 'mean'),
    ).melt(id_vars='churned', var_name='factor', value_name='average')
    st.plotly_chart(px.bar(factors, x='factor', y='average', color='churned', barmode='group',
                           title='Tenure, claims, complaints, and bundling'), use_container_width=True)

elif page == 'Customer Segments':
    def segments():
        profile = load_csv('segment_profiles.csv')
        st.subheader('K-means customer segments')
        st.dataframe(profile.style.format({'lapse_rate': '{:.1%}', 'bundled_rate': '{:.1%}', 'autopay_rate': '{:.1%}'}),
                     hide_index=True, use_container_width=True)
        st.plotly_chart(px.scatter(profile, x='avg_tenure_years', y='avg_premium_change_pct', size='customers',
                                   color='segment_name', hover_data=['lapse_rate', 'avg_late_payments'],
                                   title='Segment positioning'), use_container_width=True)
        st.caption('Segments use standardized structured behavioral and policy features. Names are analyst-friendly descriptions of the synthetic cluster profiles.')
    artifact_page(segments)

elif page == 'Text Analytics':
    def text_analytics():
        terms = load_csv('top_terms.csv')
        topics = load_csv('topics.csv')
        importance = load_csv('text_feature_importance.csv')
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(terms.head(20).sort_values('count'), x='count', y='term', orientation='h',
                               title='Most frequent service-note phrases'), use_container_width=True)
        c2.plotly_chart(px.bar(importance.head(20).sort_values('coefficient'), x='coefficient', y='feature',
                               orientation='h', title='Words and phrases most predictive of lapse'), use_container_width=True)
        st.subheader('NMF service-note topics')
        st.dataframe(topics, hide_index=True, use_container_width=True)
    artifact_page(text_analytics)

elif page == 'Survival Analysis':
    def survival():
        km = load_csv('kaplan_meier.csv')
        metrics = load_json('metrics.json')
        st.metric('Time-to-lapse MAE', f"{metrics['time_to_event']['mae_months']:.2f} months")
        st.plotly_chart(px.line(km, x='time_months', y='survival_probability',
                                title='Kaplan-Meier retention curve'), use_container_width=True)
        st.caption('The timing model is a random-forest regressor trained on observed synthetic lapse events; the curve summarizes observed retention. Censoring is not modeled beyond the event flag.')
    artifact_page(survival)

elif page == 'Monitoring & Drift':
    def monitoring():
        report = drift_report(df)
        st.subheader('Feature-distribution monitoring')
        st.dataframe(report.style.format({'baseline_mean': '{:.2f}', 'current_mean': '{:.2f}', 'standardized_shift': '{:.3f}'}),
                     hide_index=True, use_container_width=True)
        status_colors = {'Stable': '#D84A8C', 'Review': '#6E315D'}
        chart = go.Figure()
        for status, subset in report.groupby('status', sort=False):
            chart.add_trace(go.Bar(
                x=subset['feature'],
                y=subset['standardized_shift'],
                name=status,
                marker_color=status_colors.get(status, '#A82B68'),
                hovertemplate=(
                    '<b>%{x}</b><br>Standardized shift: %{y:.3f}'
                    '<br>Status: ' + status + '<extra></extra>'
                ),
            ))
            chart.add_trace(go.Scatter(
                x=subset['feature'],
                y=subset['standardized_shift'],
                mode='markers',
                showlegend=False,
                marker={'size': 10, 'color': status_colors.get(status, '#A82B68'),
                        'line': {'color': '#FFF8FC', 'width': 2}},
                hoverinfo='skip',
            ))
        chart.add_hline(y=0.20, line_dash='dash', line_color='#6E315D',
                        annotation_text='Review threshold: 0.20', annotation_position='top left')
        chart.update_layout(
            title='Structured and text feature drift against training baseline',
            yaxis={'title': 'Standardized mean shift', 'range': [-0.015, max(0.25, report['standardized_shift'].max() * 1.25)]},
            xaxis={'title': None, 'tickangle': -35},
            legend_title_text='Status',
        )
        st.plotly_chart(chart, use_container_width=True)
        st.caption('Review is triggered at a 0.20 standardized mean shift. This synthetic demonstration uses a training-profile baseline; production monitoring should compare versioned live windows and include calibration checks.')
    artifact_page(monitoring)

else:
    def claims_intelligence():
        claims, metrics = claims_summary()
        st.subheader('Claims & fraud intelligence')
        st.caption(
            'This separate module uses the user-provided claims dataset. '
            'It is distinct from the synthetic policy-lapse analytics elsewhere in PolicyGuard.'
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Claims records', f"{metrics['records']:,}")
        c2.metric('Flagged fraud rate', f"{metrics['positive_rate']:.1%}")
        c3.metric('Fraud model ROC-AUC', f"{metrics['roc_auc']:.3f}")
        c4.metric('Fraud model F1', f"{metrics['f1']:.3f}")
        left, right = st.columns(2)
        left.plotly_chart(
            px.histogram(claims, x='Claim_Type', color='Fraud_Flag', barmode='group',
                         title='Fraud flags by claim type'),
            use_container_width=True,
        )
        right.plotly_chart(
            px.box(claims, x='Fraud_Flag', y='Claim_Amount', color='Fraud_Flag',
                   title='Claim amount by fraud flag'),
            use_container_width=True,
        )
        st.subheader('Most influential non-leaking model features')
        importance = load_csv('claims_feature_importance.csv').copy()
        importance['feature'] = importance['feature'].str.replace(
            'numeric__', '', regex=False
        ).str.replace('categorical__', '', regex=False)
        st.plotly_chart(
            px.bar(importance.head(15).sort_values('coefficient'), x='coefficient', y='feature',
                   orientation='h', title='Fraud classifier coefficient importance'),
            use_container_width=True,
        )
        st.caption(
            'The classifier intentionally excludes the supplied Fraud_Risk_Score and Claim_Status '
            'to avoid target leakage. Feature coefficients are directional associations, not causal explanations.'
        )
    artifact_page(claims_intelligence)
