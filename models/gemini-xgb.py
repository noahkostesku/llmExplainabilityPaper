"""
===============================================================================
Gemini 2.5 XGB Loan Default Explanation and Evaluation Pipeline
===============================================================================

This module implements a comprehensive evaluation framework for loan default predictions using XGBoost 
models with Gemini 2.5 language model explanations. The system generates human-readable 
explanations of individual borrower risk assessments and evaluates explanation quality through 
automated scoring metrics including perplexity analysis and LLM-based grading.

The pipeline focuses on individual borrower characteristics (FICO scores, debt-to-income ratios, 
loan-to-value ratios, etc.) and generates structured explanations that prioritize feature alignment 
and directional accuracy in risk assessment.

Note: This script is intended for academic reference only.
"""

import pandas as pd
import numpy as np
import joblib

import scipy.stats as stats

import shap

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from openai import OpenAI
import openai
import google.generativeai as genai

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

from dotenv import load_dotenv

import random
import math
import json
import time
import os

GLOBAL_SYSTEM_PROMPT = """
You are a knowledgeable financial advisor explaining a credit decision to someone who wants to understand how lending works.

Think like you're a helpful banking representative speaking with a customer who is financially literate but may not know all the industry details.

Your voice should be:
- Professional yet approachable (like a trusted advisor)
- Clear and educational without being condescending 
- Honest and transparent about both positive and challenging aspects
- Focused on actionable guidance they can implement

Guidelines:
- Use accessible language like "monthly payments," "credit history," "down payment" instead of technical jargon
- Provide meaningful comparisons like "better than most applicants in your situation"
- NEVER mention: machine learning models, algorithms, technical processes, or internal bank systems
- Always explain WHY each factor matters for lending decisions
- End with a clear summary of the overall assessment without providing specific financial advice
"""

load_dotenv()

def initialize_proxies():
    
    http_proxy = os.getenv("HTTP_PROXY", "")
    https_proxy = os.getenv("HTTPS_PROXY", "")
    
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
        print(f"HTTP_PROXY set to: {http_proxy}")
    else:
        print("HTTP_PROXY not set in environment")
    
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
        print(f"HTTPS_PROXY set to: {https_proxy}")
    else:
        print("HTTPS_PROXY not set in environment")

load_dotenv()
initialize_proxies()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_GRADER_MODEL = 'gpt-4o'

EVAL_SAVE_DIR = '../evaluations/gemini_xgb/'
os.makedirs(EVAL_SAVE_DIR, exist_ok=True)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,))
)
def get_gemini_response(system_prompt, user_prompt):
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY")
        }
        
        proxies = {k: v for k, v in proxies.items() if v}
        
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.4,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 4000
            }
        }

        headers = {"Content-Type": "application/json"}
        
        response = requests.post(gemini_url, json=payload, headers=headers, proxies=proxies, timeout=30)

        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    text_response = result['candidates'][0]['content']['parts'][0]['text']
                    return text_response

    except Exception as e:
        print(f"Gemini REST API exception: {type(e).__name__}: {e}")
        return None

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError))
)
def openai_chat_completion(**kwargs):

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(**kwargs)
        return response
    except (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError) as e:
        print(f"Retryable error calling OpenAI API: {e}")
        raise
    except Exception as e:
        print(f"Non-retryable error calling OpenAI API: {e}")
        raise

def calculate_perplexity_with_gpt_neo(text, model_name="local_models/gpt-neo-125M"):
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import math

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        return perplexity

    except Exception as e:
        print(f"GPT-Neo perplexity calculation error: {e}")
        return None
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError, json.JSONDecodeError, ValueError))
)
def llm_grade_explanation(explanation, insight, model_type="xgb"):

    top_features = list(insight['top_features_impact'].keys())
    feature_impacts = []
    for feature, impact in insight['top_features_impact'].items():
        direction = "increases default risk" if impact > 0 else "decreases default risk"
        feature_impacts.append(f"{feature}: {direction} (importance: {impact:.3f})")
        
    predicted_prob = insight.get('predicted_proba', 0.5)
    approval_prob = 1 - predicted_prob

    model_context = """For XGB models: The explanation should focus primarily on individual borrower characteristics 
and feature importance. XGB models capture feature interactions but don't explicitly model network relationships.
Network coverage may be minimal or absent, which is expected for this model type."""

    prompt = f"""You are evaluating an AI explanation for loan default prediction. The explanation comes from a {model_type.upper()} model.

EXPLANATION TO EVALUATE:
"{explanation}"

GROUND TRUTH MODEL DATA:
- Model type: {model_type.upper()}
- Predicted approval probability: {approval_prob:.3f} ({approval_prob:.0%})
- Top individual features: {', '.join(top_features)}
- Feature impacts: {'; '.join(feature_impacts)}

{model_context}

EVALUATION CRITERIA:

**Individual Feature Coverage (1-5):**
Evaluates coverage of the most important SHAP features (customer-friendly terms acceptable).
- 5: Mentions all top 3 most important model features meaningfully (e.g., "credit score" for "fico")
- 4: Covers 2 of top 3 important features with good detail, minor gaps
- 3: Covers some important features but misses key ones or includes less important ones  
- 2: Limited coverage of actually important model factors, focuses on secondary features
- 1: Focuses on irrelevant factors, ignores key model learnings

**Individual Feature Consistency (1-5):**
Evaluates whether individual feature impacts are correctly described in terms of approval chances.
- 5: All mentioned individual factors have correct directional impact on approval probability
- 4: Most individual factors directionally correct, minor inconsistencies or ambiguous language
- 3: Some correct directions but notable contradictions or unclear statements
- 2: Frequent contradictions with actual model feature impacts  
- 1: Systematic misalignment with model learnings, mostly incorrect directions

**Network Coverage (1-5):**
Evaluates discussion of network/relational factors and connections. For XGB models, expect minimal network content.
- 5: Substantial network discussion with specific details (borrower counts, connection types, area patterns)
- 4: Good network focus with adequate detail about connections or area factors
- 3: Moderate network content - mentions connections but lacks depth or specificity
- 2: Limited network discussion - brief mentions without meaningful detail
- 1: No meaningful network or relational content

**Network Consistency (1-5):**
Evaluates accuracy of network relationship descriptions and risk implications. For XGB models, may not apply.
- 5: Accurate and logical network relationship descriptions with clear risk implications
- 4: Generally correct network interpretations with sound reasoning
- 3: Basic network understanding with mostly logical explanations
- 2: Some network logic but with unclear or inconsistent elements
- 1: Incorrect or contradictory network relationship descriptions

**IMPORTANT NOTES:**
- Customer-friendly language (e.g., "credit score" instead of "fico") is acceptable and preferred
- For XGB models, low network scores are expected and acceptable
- Individual feature metrics are most important for XGB model evaluation

RESPONSE FORMAT:
{{
    "Individual Feature Coverage": <score>,
    "Individual Feature Consistency": <score>,
    "Network Coverage": <score>,
    "Network Consistency": <score>
}}"""

    system_msg = {
        "role": "system",
        "content": "You are an expert evaluator trained to assess different types of ML explanations appropriately. You must respond **only** with a valid JSON object."
    }
    user_msg = {"role": "user", "content": prompt}

    response = openai_chat_completion(
        model=OPENAI_GRADER_MODEL,
        messages=[system_msg, user_msg],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=200
    )

    try:
        grades = json.loads(response.choices[0].message.content)

        expected_keys = ["Individual Feature Coverage", "Individual Feature Consistency", 
                        "Network Coverage", "Network Consistency"]

        for key, score in grades.items():
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                print(f"Invalid score for {key}: {score}")
                raise ValueError(f"Invalid score range for {key}")

        grades = {key: int(score) for key, score in grades.items()}
        
        print(f"\n### {model_type.upper()} Model LLM Grading Results ###")
        for k, v in grades.items():
            print(f"{k}: {v}/5")
            
        return grades

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}. Content: {response.choices[0].message.content}")
        raise
    except ValueError as e:
        print(f"Validation error in grader response: {e}")
        raise

def optimize_f1_threshold(model, X_test, y_test, eval_save_dir_dummy="."):

    from sklearn.metrics import precision_recall_curve, classification_report
    
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\n### F1-score Optimization ###")
    print(f"Optimized F1-score: {best_f1:.4f} at threshold: {best_threshold:.4f}")

    y_pred_optimized = (y_probs >= best_threshold).astype(int)
    print("\n### Classification Report at Optimized Threshold ###")
    print(classification_report(y_test, y_pred_optimized, target_names=['No Default', 'Default']))

    return best_threshold, best_f1

def export_shap_insights_for_single_borrower(model, X_data, y_data, specific_idx, optimized_threshold, num_top_features=5):

    explainer = shap.TreeExplainer(model)
    
    single_instance = X_data.loc[[specific_idx]]
    single_prediction_shap_values = explainer.shap_values(single_instance)

    if isinstance(single_prediction_shap_values, list):
        single_prediction_shap_values = single_prediction_shap_values[1]

    actual_feature_values = {}
    borrower_row = X_data.loc[specific_idx]
    for feature in X_data.columns:
        actual_feature_values[feature] = float(borrower_row[feature])

    single_shap_dict = {
        'index': int(specific_idx),
        'predicted_proba': float(model.predict_proba(single_instance)[:, 1][0]),
        'actual_label': int(y_data.loc[specific_idx]),
        'top_features_impact': {},
        'actual_feature_values': actual_feature_values
    }

    feature_impact = pd.DataFrame({
        'feature': X_data.columns,
        'shap_value': single_prediction_shap_values[0]
    }).sort_values(by='shap_value', key=abs, ascending=False)

    for _, row in feature_impact.head(num_top_features).iterrows():
        single_shap_dict['top_features_impact'][row['feature']] = float(row['shap_value'])

    prediction_outcome = "Default" if single_shap_dict['predicted_proba'] >= optimized_threshold else "No Default"
    actual_outcome = "Default" if single_shap_dict['actual_label'] == 1 else "No Default"

    print(f"\n### Selected Borrower SHAP Insights (Index: {specific_idx}) ###\n")
    print(f"Predicted Outcome: {prediction_outcome} (Probability: {single_shap_dict['predicted_proba']:.4f})")
    print(f"Actual Outcome: {actual_outcome}")
    print(f"Top {num_top_features} Contributing Factors:")
    for feat, val in single_shap_dict['top_features_impact'].items():
        direction = "towards default" if val > 0 else "towards no default"
        print(f"{feat}: {val:.4f} (pushes {direction})")

    return single_shap_dict

def build_gemini_prompt_for_single_borrower(borrower_shap_insight, optimized_threshold, X_test=None, borrower_idx=None):

    predicted_outcome = 'Default' if borrower_shap_insight['predicted_proba'] >= optimized_threshold else 'No Default'
    actual_outcome = 'Default' if borrower_shap_insight['actual_label'] == 1 else 'No Default'
    
    df_row = None
    if X_test is not None and borrower_idx is not None:
        df_row = X_test.loc[borrower_idx]
    
    credit_score_explanation = _get_credit_score_explanation(df_row, df_row.get('fico', 0.5))
    
    top_shap_features = sorted(
        borrower_shap_insight['top_features_impact'].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    customer_factors = []
    improvement_advice = []
    top_factors_formatted = [f"1. {credit_score_explanation}"]
    
    factor_count = 2 
    for feature, shap_impact in top_shap_features:
        if feature == 'fico':
            continue
            
        feature_value = df_row.get(feature, 'N/A')
        plain_name, pop_context, impact_level, direction, advice = _translate_features_to_customer_language(
            feature, feature_value, shap_impact
        )
        
        customer_factors.append(f"your {plain_name} {impact_level} {direction} because it {pop_context}")
        top_factors_formatted.append(f"{factor_count}. Your {plain_name} {pop_context} and {impact_level} {direction}")
        
        if "hurts" in direction and impact_level in ["really", "significantly"]:
            improvement_advice.append(advice)
            
        factor_count += 1
        if factor_count > 4:
            break
    
    factors_text = ", and ".join(customer_factors[:3])
    advice_text = improvement_advice[0] if improvement_advice else "keep doing what you're doing financially"
    top_factors_list = "\n".join(top_factors_formatted[:4])
    
    approval_chance = (1 - borrower_shap_insight['predicted_proba']) * 100
    if approval_chance > 85:
        approval_context = f"excellent approval chances ({approval_chance:.0f}% likely to be approved)"
    elif approval_chance > 65:
        approval_context = f"good approval chances ({approval_chance:.0f}% likely to be approved)"
    elif approval_chance > 50:
        approval_context = f"fair approval chances ({approval_chance:.0f}% likely to be approved)"
    else:
        approval_context = f"challenging approval odds ({approval_chance:.0f}% likely to be approved)"
    
    system_prompt = GLOBAL_SYSTEM_PROMPT
    
    user_prompt = f"""
You are explaining to someone why their loan application has {approval_context}.

{credit_score_explanation}.

SITUATION SUMMARY:
{factors_text}.

KEY FACTORS (most important first):
{top_factors_list}

WRITE AN EXPLANATION that a smart 17-year-old would understand. Follow these rules:

1. Start by explaining what their {approval_chance:.0f}% approval chance means compared to other people
2. Always mention their credit score situation first
3. Explain the next most important factor in simple terms and why it matters to banks
4. Mention how these factors work together  
5. Give ONE specific action they can take: {advice_text}
6. End with encouragement about their overall situation

REQUIREMENTS:
- Never use banking terms like "loan-to-value" or "debt-to-income" 
- Always compare to "most people" or "other borrowers"
- Give specific timelines like "in 6 months" or "over the next year"
- Explain WHY each factor matters to banks
- Sound like a helpful teacher, not a banker
- Keep to 6-7 sentences total

TONE: Helpful and encouraging, like explaining to a smart teenager who wants to understand how money works.
"""
    
    return system_prompt, user_prompt

def compute_confidence_interval(scores, confidence=0.95):

    mean_score = np.mean(scores)
    
    if np.std(scores) == 0:
        return mean_score, f"{mean_score:.2f} ± 0.00 (95% CI = [{mean_score:.2f}, {mean_score:.2f}]) - identical values"
    
    std_err = stats.sem(scores)
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(scores) - 1)
    ci_low = mean_score - h
    ci_high = mean_score + h
    ci_str = f"{mean_score:.2f} ± {h:.2f} (95% CI = [{ci_low:.2f}, {ci_high:.2f}])"
    
    return mean_score, ci_str

def _get_credit_score_explanation(df_row, feature_value):

    if 'fico_actual' in df_row.index and not pd.isna(df_row['fico_actual']):
        actual_score = int(df_row['fico_actual'])
    elif 'fico' in df_row.index:
        normalized_fico = df_row['fico']
        actual_score = int(300 + (normalized_fico * 550))
    else:
        return "Your credit score is about average compared to other borrowers"
    
    if actual_score >= 740:
        score_quality = "excellent"
    elif actual_score >= 670:
        score_quality = "good" 
    elif actual_score >= 580:
        score_quality = "fair"
    elif actual_score >= 500:
        score_quality = "below average"
    else:
        score_quality = "poor"
        
    return f"Your credit score of {actual_score} is {score_quality} compared to most borrowers"

def _translate_features_to_customer_language(feature, feature_value, shap_value):

    feature_translations = {
        'fico': {
            'name': 'credit score',
            'ranges': [(0, 0.2, "is much lower than most people (bottom 20%)"), 
                      (0.2, 0.4, "is below average (bottom 40%)"), 
                      (0.4, 0.6, "is about average (middle 20%)"), 
                      (0.6, 0.8, "is above average (top 40%)"), 
                      (0.8, 1.0, "is excellent (top 20%)")],
            'advice': "pay all bills on time for 6 months and keep credit card balances under 30% to improve your score"
        },
        'if_fthb': {
            'name': 'first-time buyer status',
            'ranges': [(0, 0.5, "means you've bought homes before (experienced buyer)"),
                      (0.5, 1, "means this is your first home purchase")],
            'advice': "first-time buyers often get special programs and lower down payment options"
        },
        'cnt_borr': {
            'name': 'number of borrowers',
            'ranges': [(1, 1.5, "means you're applying solo"),
                      (1.5, 2.5, "means you're applying with one other person"),
                      (2.5, 100, "means you have multiple co-borrowers")],
            'advice': "having a co-borrower can help if they have good credit and income"
        },
        'cnt_units': {
            'name': 'property units',
            'ranges': [(1, 1.5, "means it's a single-family home"),
                      (1.5, 2.5, "means it's a duplex property"),
                      (2.5, 100, "means it's a multi-unit property")],
            'advice': "single-family homes typically have the best loan terms and rates"
        },
        'dti': {
            'name': 'debt-to-income ratio',
            'ranges': [(0, 0.2, "is excellent - your monthly debts are much lower than most people (bottom 20%)"), 
                      (0.2, 0.4, "is good - your monthly debts are below average (bottom 40%)"), 
                      (0.4, 0.6, "is typical - about average debt levels (middle 20%)"), 
                      (0.6, 0.8, "is concerning - higher monthly debts than most borrowers (top 40%)"), 
                      (0.8, 1.0, "is very high - much more monthly debt than banks prefer (top 20%)")],
            'advice': "pay down credit cards and other debts over the next 6-12 months to improve this ratio"
        },
        'ltv': {
            'name': 'down payment amount',
            'ranges': [(0, 0.2, "means you're putting down much more than most people (top 20% down payment)"), 
                      (0.2, 0.4, "means you're putting down more than average (above average down payment)"), 
                      (0.4, 0.6, "means your down payment is typical (middle 20%)"), 
                      (0.6, 0.8, "means you're putting down less than most people (below average)"), 
                      (0.8, 1.0, "means you're putting down very little (bottom 20%)")],
            'advice': "try to save for a larger down payment - even 5% more can significantly improve your terms"
        },
        'orig_upb': {
            'name': 'loan amount',
            'ranges': [(0, 0.2, "is much smaller than most loans (bottom 20% - very manageable)"),
                      (0.2, 0.4, "is below average loan size (bottom 40% - good for your budget)"),
                      (0.4, 0.6, "is a typical loan amount (middle 20%)"),
                      (0.6, 0.8, "is larger than most people borrow (top 40%)"),
                      (0.8, 1.0, "is much larger than typical loans (top 20%)")],
            'advice': "consider a less expensive home to reduce your monthly payments and improve approval odds"
        },
        'loan_term': {
            'name': 'loan term length',
            'ranges': [(0, 0.2, "is much shorter than most people choose (bottom 20% - saves money long-term)"),
                      (0.2, 0.4, "is shorter than average (below average - good for saving interest)"),
                      (0.4, 0.6, "is a typical loan length (middle 20%)"),
                      (0.6, 0.8, "is longer than most people choose (top 40%)"),
                      (0.8, 1.0, "is much longer than typical loans (top 20%)")],
            'advice': "consider a standard 30-year term to lower monthly payments if needed"
        },
        'if_prim_res': {
            'name': 'primary residence status',
            'ranges': [(0, 0.5, "means this is an investment or vacation property"),
                      (0.5, 1, "means this will be your main home")],
            'advice': "primary residences get the best rates and terms from lenders"
        },
        'if_corr': {
            'name': 'correspondent lending status',
            'ranges': [(0, 0.5, "means you're working with a direct lender"),
                      (0.5, 1, "means you're working through a correspondent lender")],
            'advice': "different lender types can offer different rates - shop around to compare"
        },
        'if_sf': {
            'name': 'single-family property type',
            'ranges': [(0, 0.5, "means you're buying a condo, townhome, or other property type"),
                      (0.5, 1, "means you're buying a single-family detached home")],
            'advice': "single-family homes typically qualify for the best loan programs and rates"
        },
        'if_purc': {
            'name': 'purchase vs refinance',
            'ranges': [(0, 0.5, "means you're refinancing an existing loan"),
                      (0.5, 1, "means you're purchasing a new home")],
            'advice': "purchases often have more loan program options than refinances"
        }
    }
    
    if feature not in feature_translations:
        return f"your {feature.replace('_', ' ')}", "is about average", "somewhat", "affects your approval", "talking to a loan officer about this"
    
    trans = feature_translations[feature]
    plain_name = trans['name']

    population_context = "is about average compared to other borrowers"
    for min_val, max_val, description in trans['ranges']:
        if min_val <= feature_value <= max_val:
            population_context = description
            break
    
    abs_shap = abs(shap_value)
    if abs_shap > 0.15:
        impact_level = "really"
    elif abs_shap > 0.08:
        impact_level = "significantly"  
    elif abs_shap > 0.03:
        impact_level = "somewhat"
    else:
        impact_level = "slightly"
    
    direction = "hurts your approval chances" if shap_value > 0 else "helps your approval chances"
    
    return plain_name, population_context, impact_level, direction, trans['advice']

def evaluate_explanations_with_scoring(model, X_test, y_test, optimized_threshold, selected_indices=None, num_samples=10):
    
    print(f"\n### Starting Evaluation for {len(selected_indices) if selected_indices else num_samples} samples ### ")

    explanations_dir = "explanations"
    os.makedirs(explanations_dir, exist_ok=True)
    explanations_file = os.path.join(explanations_dir, "gemini_xgb_explanations.jsonl")

    if selected_indices is None:
        selected_indices = random.sample(X_test.index.tolist(), min(num_samples, len(X_test)))
    
    results = []
    openai_perplexity_scores = []
    
    grading_scores_by_dimension = {
        "Individual Feature Coverage": [],
        "Individual Feature Consistency": [],
        "Network Coverage": [],
        "Network Consistency": []
    }
    
    successful_evaluations = 0
    
    for i, idx in enumerate(selected_indices):
        print(f"\nProcessing sample {i+1}/{len(selected_indices)} (Index: {idx})")
        
        try:
            insight = export_shap_insights_for_single_borrower(
                model, X_test, y_test, idx, optimized_threshold=optimized_threshold
            )
            
            system_prompt, user_prompt = build_gemini_prompt_for_single_borrower(
                insight, optimized_threshold, X_test=X_test, borrower_idx=idx
            )
            explanation = get_gemini_response(system_prompt, user_prompt)
                
            print(f"\nGenerated Explanation: \"{explanation}...\"")
            
            entry = {
                "node_index": int(idx),
                "explanation": explanation
            }
            with open(explanations_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            openai_perplexity = calculate_perplexity_with_gpt_neo(explanation)
                
            if openai_perplexity is not None:
                openai_perplexity_scores.append(openai_perplexity)
                print(f"Perplexity: {openai_perplexity:.2f}")

            grades = llm_grade_explanation(explanation, insight, model_type="xgb")
            if grades is not None:
                for dimension, score in grades.items():
                    if dimension in grading_scores_by_dimension:
                        grading_scores_by_dimension[dimension].append(score)

                print(f"LLM Grades: {grades}")
                successful_evaluations += 1
                
            results.append({
                'index': idx,
                'explanation': explanation,
                'openai_perplexity': openai_perplexity,
                'llm_grades': grades,
                'insight': insight
            })
            
        except Exception as e:
            print(f"Critical error processing sample {idx}: {e}")
            continue
    
    print(f"EVALUATION COMPLETE: {successful_evaluations}/{len(selected_indices)} samples processed successfully")
    print(f"Explanations saved to: {explanations_file}")
    
    results_path = os.path.join(EVAL_SAVE_DIR, 'explanation_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")
    
    evaluation_stats = {
        'successful_evaluations': successful_evaluations,
        'total_samples': len(selected_indices),
        'perplexity_scores': openai_perplexity_scores,
        'grading_scores': grading_scores_by_dimension,
        'explanations_file': explanations_file,
        'results_file': results_path
    }
    
    return results, evaluation_stats

if __name__ == "__main__":
    
    print("\nStarting Explanation Evaluation Workflow\n")
    
    MODEL_FILE_NAME = 'trained_xgb_model.pkl'
    MODEL_SAVE_DIR = "./"
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILE_NAME)
    model = joblib.load(MODEL_PATH)

    try:
        test_data_path = '../data/cleaned_data/df_origination_test_scaled.csv'
        test = pd.read_csv(test_data_path)
        exclude_cols = ['id', 'id_loan', 'year', 'month', 'provider', 'area', 'svcg_cycle']

        if 'd_timer' in test.columns:
            exclude_cols.append('d_timer')

        cols_to_drop_from_test = [col for col in exclude_cols if col in test.columns]
        X_test = test.drop(columns=cols_to_drop_from_test + [target_col])
        y_test = test[target_col]
        print(f"Successfully loaded test data. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
 
    except Exception as e:
        print(f"Error loading or processing test data: {e}")
        exit()

    optimized_threshold, _ = optimize_f1_threshold(model, X_test, y_test)
    print(f"Using optimized threshold: {optimized_threshold:.4f}")

    random.seed(42)
    num_samples = 100
    selected_indices = random.sample(X_test.index.tolist(), min(num_samples, len(X_test)))
    
    indices_save_path = os.path.join(EVAL_SAVE_DIR, 'selected_borrower_indices.json')
    with open(indices_save_path, 'w') as f:
        json.dump(selected_indices, f)
    print(f"Selected indices saved to: {indices_save_path}")

    results, evaluation_stats = evaluate_explanations_with_scoring(
        model, X_test, y_test, optimized_threshold, selected_indices=selected_indices
    )
    
    print("\n ### RESULTS ###\n")
    
    if evaluation_stats['perplexity_scores']:
        openai_perp_mean, openai_perp_ci_str = compute_confidence_interval(
            evaluation_stats['perplexity_scores']
        )
        print(f"\nPerplexity:")
        print(f"{openai_perp_ci_str}")
        print(f"Samples: {len(evaluation_stats['perplexity_scores'])}")
    
    print(f"\nLLM Grading Results:")
    for dimension, scores in evaluation_stats['grading_scores'].items():
        if scores:
            dim_mean, dim_ci_str = compute_confidence_interval(scores)
            print(f"{dimension}:")
            print(f"{dim_ci_str}/5")
        
    print("\nEvaluation Complete!")