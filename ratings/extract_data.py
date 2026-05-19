#!/usr/bin/env python3
import json
import csv
from pathlib import Path

EXPLANATIONS_DIR = Path('../models/explanations')
GRADES_DIR = Path('../models/grades')
OUT_DIR = Path('.')

PIPELINE_MAP = {'xgb':'Tabular-based','gnn':'Network-based','hybrid': 'Bimodal',}
LLM_MAP = {'gemma3':'Gemma 3 (ft)','deepseek-r1-70b':'DeepSeek R1 (ft)','gemini':'Gemini 2.5',
}
METRIC_MAP = {'Individual Feature Coverage':'TFC','Individual Feature Consistency': 'TDC','Network Coverage':'NFC','Network Consistency':'NDC',}
FILES = [
    'gemma-xgb-explanations.jsonl',
    'gemma-gat-explanations.jsonl',
    'gemma-hybrid-explanations.jsonl',
    'deepseek-xgb-explanations.jsonl',
    'deepseek-gnn-explanations.jsonl',
    'deepseek-hybrid-explanations.jsonl',
    'gemini-xgb-explanations.json',
    'gemini-gat-explanations.jsonl',
    'gemini-hybrid-explanations.jsonl',
]


def load_file(path: Path):
    content = path.read_text().strip()
    if content.startswith('['):
        return json.loads(content)
    return [json.loads(l) for l in content.splitlines() if l.strip()]


def main():
    perplexity_rows = []
    judge_rows = []
    
    for fname in FILES:
        path = EXPLANATIONS_DIR / fname
        records = load_file(path)
        sample = records[0]
        pipeline = PIPELINE_MAP.get(sample.get('model_type', ''))
        llm = LLM_MAP.get(sample.get('llm_name', ''))
        perp_count = 0
        for rec in records:
            perp = rec.get('perplexity')
            if perp is not None:
                perplexity_rows.append({'pipeline': pipeline, 'llm': llm, 'perplexity': perp})
                perp_count += 1

        print(f'{fname}: {len(records)} recs — {perp_count} perplexity')

    GRADE_LLM_MAP = {'gemma':'Gemma 3 (ft)','deepseek':'DeepSeek R1 (ft)','gemini':'Gemini 2.5',}
    GRADE_PIPELINE_MAP = {'tabular': 'Tabular-based','network':'Network-based','hybrid':'Bimodal',}
    for grade_file in sorted(GRADES_DIR.glob('*_grades.jsonl')):
        parts = grade_file.stem.replace('_grades', '').rsplit('_', 1)
        if len(parts) != 2:
            continue
        llm_key, pipeline_key = parts
        pipeline = GRADE_PIPELINE_MAP.get(pipeline_key)
        llm = GRADE_LLM_MAP.get(llm_key)
        grade_recs = [json.loads(l) for l in grade_file.read_text().splitlines() if l.strip()]
        
        for rec in grade_recs:
            for key, metric in METRIC_MAP.items():
                val = rec.get(key)
                if val not in (0, None):
                    judge_rows.append({'pipeline': pipeline, 'llm': llm, 'metric': metric, 'score': val})

        print(f'{grade_file.name}: {len(grade_recs)} graded records')

    perp_path = OUT_DIR / 'perplexity_data.csv'
    with open(perp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pipeline', 'llm', 'perplexity'])
        w.writeheader()
        w.writerows(perplexity_rows)
    print(f'\nWrote {len(perplexity_rows)} rows -> {perp_path}')

    judge_path = OUT_DIR / 'llm_judge_data.csv'
    with open(judge_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pipeline', 'llm', 'metric', 'score'])
        w.writeheader()
        w.writerows(judge_rows)
    print(f'Wrote {len(judge_rows)} rows -> {judge_path}')


if __name__ == '__main__':
    main()
