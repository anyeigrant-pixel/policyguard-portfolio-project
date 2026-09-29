import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import DATA_DIR

RNG = np.random.default_rng(42)
N = 10000

def make_note(row):
    notes = []
    if row['premium_change_pct'] > 10:
        notes.append('Customer is frustrated about a recent premium increase')
    if row['late_payments_12m'] >= 2:
        notes.append('Customer discussed repeated payment timing issues')
    if row['service_tickets_6m'] >= 3:
        notes.append('Customer reported repeated service delays and requested escalation')
    if row['claims_3y'] >= 2:
        notes.append('Customer asked about how recent claims affected renewal pricing')
    if row['bundled_policies'] == 0:
        notes.append('Customer currently holds a single policy and asked about alternatives')
    if row['tenure_years'] < 2:
        notes.append('Newer customer requested clarification on coverage and billing')
    if not notes:
        notes.append('Customer completed a routine policy service interaction with no major concerns')
    if row['churned'] == 1 and RNG.random() < 0.5:
        notes.append('Customer mentioned shopping for a different insurer before renewal')
    return '. '.join(notes) + '.'

def main():
    df = pd.DataFrame({
        'customer_id': np.arange(100001, 100001+N),
        'age': RNG.integers(21, 76, N),
        'tenure_years': np.round(RNG.uniform(0.2, 18, N), 1),
        'annual_premium': np.round(RNG.normal(1850, 550, N).clip(500, 5000), 2),
        'premium_change_pct': np.round(RNG.normal(6, 8, N).clip(-15, 35), 1),
        'late_payments_12m': RNG.poisson(0.7, N).clip(0, 6),
        'claims_3y': RNG.poisson(0.8, N).clip(0, 6),
        'service_tickets_6m': RNG.poisson(1.2, N).clip(0, 8),
        'bundled_policies': RNG.binomial(1, 0.58, N),
        'autopay': RNG.binomial(1, 0.67, N),
        'paperless': RNG.binomial(1, 0.74, N),
        'policy_type': RNG.choice(['Auto','Home','Renters'], N, p=[0.5,0.35,0.15]),
        'region': RNG.choice(['Northeast','South','Midwest','West'], N),
    })
    logit = (
        -2.2 + 0.06*df['premium_change_pct'] + 0.45*df['late_payments_12m']
        + 0.28*df['service_tickets_6m'] + 0.18*df['claims_3y']
        - 0.12*df['tenure_years'] - 0.55*df['bundled_policies']
        - 0.35*df['autopay']
    )
    prob = 1/(1+np.exp(-logit))
    df['churned'] = RNG.binomial(1, prob.clip(0.02,0.9))
    base_months = RNG.exponential(scale=24, size=N)
    hazard_mod = np.exp(
        0.025*df['premium_change_pct'] + 0.2*df['late_payments_12m']
        + 0.12*df['service_tickets_6m'] - 0.04*df['tenure_years']
        - 0.25*df['bundled_policies']
    )
    df['months_to_event'] = np.round((base_months / hazard_mod).clip(1, 60), 1)
    df['event_observed'] = df['churned']
    df['service_note'] = df.apply(make_note, axis=1)
    out = DATA_DIR / 'processed' / 'policyholders.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'wrote {len(df):,} rows to {out}')

if __name__ == '__main__':
    main()
