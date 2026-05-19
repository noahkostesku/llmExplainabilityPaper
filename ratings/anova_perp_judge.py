import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

PERPLEXITY_CSV = 'perplexity_data.csv'
JUDGE_CSV = 'llm_judge_data.csv'
OUT_DIR = './anova_simulated_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

METRIC_MAP = {'TFC': 'individual_feature_coverage', 'TDC': 'individual_feature_consistency', 'NFC': 'network_coverage', 'NDC': 'network_consistency',}

def run_two_way_anova(df: pd.DataFrame, metric: str, typ: int = 2):
    d = df.dropna(subset=[metric, 'pipeline', 'llm']).copy()
    n_pipe = d['pipeline'].nunique()
    n_llm  = d['llm'].nunique()

    print(f'\n{metric}')
    print(f'Rows: {len(d)} | pipelines: {n_pipe} | LLMs: {n_llm}')

    cell_counts = d.groupby(['pipeline', 'llm']).size().unstack(fill_value=0)
    print('  Cell counts (pipeline x llm):')
    print(cell_counts.to_string())

    formula = f'{metric} ~ C(pipeline) + C(llm) + C(pipeline):C(llm)'
    model = smf.ols(formula, data=d).fit()
    aov = anova_lm(model, typ=typ)
    aov['eta_sq'] = aov['sum_sq'] / aov['sum_sq'].sum()
    return aov

perp_df = pd.read_csv(PERPLEXITY_CSV)
print(f'Perplexity: {len(perp_df)} rows')
print(f'Pipelines: {sorted(perp_df["pipeline"].unique())}')
print(f'LLMs:{sorted(perp_df["llm"].unique())}')

aov = run_two_way_anova(perp_df, 'perplexity')
if aov is not None:
    print(aov.round(4))
    aov.to_csv(os.path.join(OUT_DIR, 'anova_perplexity.csv'))

judge_df = pd.read_csv(JUDGE_CSV)
print(f'\nJudge scores: {len(judge_df)} rows')
print(f'Metrics: {sorted(judge_df["metric"].unique())}')

for code, col in METRIC_MAP.items():
    subset = judge_df[judge_df['metric'] == code].copy()
    subset = subset.rename(columns={'score': col})
    aov = run_two_way_anova(subset, col)
    if aov is not None:
        print(aov.round(4))
        aov.to_csv(os.path.join(OUT_DIR, f'anova_{col}.csv'))

print(f'\nAll ANOVA tables saved to: {os.path.abspath(OUT_DIR)}')
