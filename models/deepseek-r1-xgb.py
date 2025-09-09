"""
===================================================================================
DeepSeek-R1 70B Fine-tuned XGBoost-Based Loan Default Explanation and Evaluation Script
===================================================================================

This file implements a hybrid machine learning explanation pipeline combining 
an XGBoost classifier for tabular loan default prediction with a fine-tuned 
DeepSeek-R1 70B language model for high-quality natural language explanations.

This pipeline is specifically tailored for evaluating model explainability in 
credit risk prediction contexts. It enables both qualitative and quantitative 
assessment of explanation quality using expert-designed scoring rubrics.

Note: This script is intended for academic reference only.

"""

import pandas as pd
import numpy as np
import joblib

import scipy.stats as stats

import shap

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from openai import OpenAI
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from dotenv import load_dotenv

from peft import PeftModel
from transformers import LogitsProcessor

import os
import random
import math
import json

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_GRADER_MODEL = 'gpt-4o'

EVAL_SAVE_DIR = '../evaluations/deepseek_xgb/'
os.makedirs(EVAL_SAVE_DIR, exist_ok=True)

class NumericalStabilityProcessor(LogitsProcessor):
    
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-100.0, posinf=100.0, neginf=-100.0)
        return torch.clamp(scores, min=-100.0, max=100.0)

class DeepSeekConfig:
    
    DEEPSEEK_MODEL_PATH = "<my-path>/deepseek-r1-70b-it-finetuned-xgb-8bit"
    BASE_MODEL_PATH = "<my-path>/DeepSeek-R1-Distill-Llama-70B"

def initialize_deepseek():

    print("Loading fine-tuned DeepSeek-R1 model...")

    try:
        config = DeepSeekConfig()

        print(f"Loading base model from: {config.BASE_MODEL_PATH}")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading LoRA adapter from: {config.DEEPSEEK_MODEL_PATH}")
        model = PeftModel.from_pretrained(
            base_model,
            config.DEEPSEEK_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16
        )

        model.eval() 
        torch.cuda.empty_cache() 

        print("DeepSeek-R1 model loaded in 8-bit with LoRA adapter")
        return model, tokenizer, True

    except Exception as e:
        print(f"Error loading DeepSeek-R1 model: {e}")
        return None, None, False

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,))
)
def get_deepseek_response(system_prompt, user_prompt, model, tokenizer):

    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8000,
            padding=True
        )

        device = next(model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        print("Generating explanation with DeepSeek-R1...")

        model.eval()
        torch.cuda.empty_cache()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=[NumericalStabilityProcessor()],
                num_return_sequences=1
            )

        generated_tokens = outputs[:, input_ids.shape[1]:]
        explanation = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        explanation = explanation.strip()

        stop_patterns = [
            "user\n",
            "GUIDED SUBGOAL STRUCTURE",
            "OUTPUT LENGTH/FORMAT CONSTRAINT",
            "BORROWER-LEVEL COUNTERFACTUAL",
            "INSTRUCTIONAL CHAIN-OF-THOUGHT",
            "\n\nmodel",
            "\nmodel"
        ]

        for pattern in stop_patterns:
            if pattern in explanation:
                explanation = explanation.split(pattern)[0].strip()
                print(f"Post-processing stopped at pattern: '{pattern}'")
                break

        print(f"Generated explanation length: {len(explanation)} characters")
        return explanation

    except Exception as e:
        print(f"Error: Could not generate explanation - {str(e)}")
        return f"Error: Could not generate explanation - {str(e)}"

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

    model_context = f"""For {model_type.upper()} models: Evaluate according to the model's intended purpose and capabilities."""

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
Evaluates discussion of network/relational factors and connections.
- 5: Substantial network discussion with specific details (borrower counts, connection types, area patterns)
- 4: Good network focus with adequate detail about connections or area factors
- 3: Moderate network content - mentions connections but lacks depth or specificity
- 2: Limited network discussion - brief mentions without meaningful detail
- 1: No meaningful network or relational content

**Network Consistency (1-5):**
Evaluates accuracy of network relationship descriptions and risk implications.
- 5: Accurate and logical network relationship descriptions with clear risk implications
- 4: Generally correct network interpretations with sound reasoning
- 3: Basic network understanding with mostly logical explanations
- 2: Some network logic but with unclear or inconsistent elements
- 1: Incorrect or contradictory network relationship descriptions

**IMPORTANT NOTES:**
- Customer-friendly language (e.g., "credit score" instead of "fico") is acceptable and preferred
- Network/relationship factors may be described as "area patterns" or "similar borrowers"
- For XGB models, network coverage will typically be low since they focus on individual features

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

def build_deepseek_prompt_for_single_borrower(borrower_shap_insight, optimized_threshold, X_test=None, borrower_idx=None):

    predicted_outcome = 'Default' if borrower_shap_insight['predicted_proba'] >= optimized_threshold else 'No Default'
    actual_outcome = 'Default' if borrower_shap_insight['actual_label'] == 1 else 'No Default'

    df_row = None
    if X_test is not None and borrower_idx is not None:
        df_row = X_test.loc[borrower_idx]

    top_shap_features = sorted(
        borrower_shap_insight['top_features_impact'].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    customer_factors = []
    top_factors_formatted = []
    factor_count = 1

    for feature, shap_impact in top_shap_features: 
        feature_value = df_row.get(feature, 'N/A') if hasattr(df_row, 'get') else borrower_shap_insight['actual_feature_values'].get(feature, 0)

        plain_name, pop_context, impact_level, direction, advice = _translate_features_to_customer_language(
            feature, feature_value, shap_impact
        )

        factor_text = f"Your {plain_name} {pop_context} and {impact_level} {direction}."
        customer_factors.append(factor_text)
        top_factors_formatted.append(f"{factor_count}. {factor_text}")
        factor_count += 1

    approval_chance = (1 - borrower_shap_insight['predicted_proba']) * 100

    if approval_chance > 85:
        approval_context = f"excellent approval chances ({approval_chance:.0f}%)"
    elif approval_chance > 65:
        approval_context = f"strong approval chances ({approval_chance:.0f}%)"
    elif approval_chance > 50:
        approval_context = f"reasonable approval chances ({approval_chance:.0f}%)"
    else:
        approval_context = f"challenging approval process ({approval_chance:.0f}%)"

    system_prompt = GLOBAL_SYSTEM_PROMPT

    user_prompt = f"""
You are explaining to someone why their loan application has {approval_context} based on their personal financial profile.

FINANCIAL ANALYSIS: {', '.join(customer_factors[:3])}

Your {approval_chance:.0f}% approval chance is determined by these personal financial factors.

WRITE A DETAILED EXPLANATION following these rules:
1. Start with what their {approval_chance:.0f}% approval chance means (2 sentences)
2. **Personal Financial Factors** – For each factor listed below, explain why it matters and how it affects loan terms (2-3 sentences per major factor):
   {chr(10).join([f"   → {factor}" for factor in top_factors_formatted])}
3. **Combined Financial Analysis** – Describe how all personal factors work together to influence approval (2-3 sentences)
4. **Actionable Improvement Advice** – Give specific steps to strengthen their financial profile (2-3 sentences)
5. **Comparison Perspective** – Compare them to other applicants in similar financial situations (2 sentences)
6. End with overall financial outlook and next steps (2 sentences)

REQUIREMENTS:
- Write 6-8 sentences for comprehensive coverage
- Cover ALL {len(top_factors_formatted)} listed financial factors - do not skip any
- Focus entirely on personal financial factors
- Do not use technical terms like "model", "algorithm", or "SHAP"
- Give concrete comparisons like "better than X% of applicants"
- Provide actionable recommendations for their individual situation
- Use clear, non-technical language throughout

TONE: Like a financial advisor explaining their individual financial assessment in detail.
"""

    return system_prompt, user_prompt

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

def evaluate_explanations_with_scoring(model, X_test, y_test, optimized_threshold, selected_indices=None, num_samples=10):

    print(f"\n### Starting Comprehensive Explanation Evaluation for {len(selected_indices) if selected_indices else num_samples} samples ### ")

    deepseek_model, deepseek_tokenizer, success = initialize_deepseek()
    if not success:
        print("Failed to initialize DeepSeek model. Exiting...")
        return [], {}

    explanations_dir = "explanations"
    os.makedirs(explanations_dir, exist_ok=True)
    explanations_file = os.path.join(explanations_dir, "deepseek_xgb_explanations.jsonl")  # Changed filename

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

            system_prompt, user_prompt = build_deepseek_prompt_for_single_borrower(
    insight, optimized_threshold, X_test=X_test, borrower_idx=idx
)
            explanation = get_deepseek_response(system_prompt, user_prompt, deepseek_model, deepseek_tokenizer)

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

    print(f"EVALUATION COMPLETE: {successful_evaluations}/{num_samples} samples processed successfully")
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

            unique_scores = list(set(scores))
            if len(unique_scores) <= 3:
                score_counts = {score: scores.count(score) for score in unique_scores}

    print("\nEvaluation Complete!")