"""
===============================================================================
Gemma 3 4B Fine-tuned Hybrid Loan Default Explanation and Evaluation Pipeline
===============================================================================

This file implements a comprehensive machine learning pipeline that combines GATs
and XGBoost models to predict loan defaults using both individual borrower characteristics and network relationship patterns. The system integrates SHAP explainability for tabular features, GNNExplainer for graph relationships, and fine-tuned Gemma 3 4B language models to generate human-readable explanations of prediction decisions.

The pipeline supports both single-node analysis and cross-sectional evaluation across multiple borrowers, with comprehensive confidence interval reporting and explanation quality assessment.

Note: This script is intended for academic reference only.
"""

import pandas as pd
import numpy as np
import joblib

import random
from sklearn.metrics import roc_auc_score

import scipy.stats as stats

import shap

import torch
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from transformers import AutoTokenizer, AutoModelForCausalLM

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from peft import PeftModel
from transformers import LogitsProcessor

from dotenv import load_dotenv

import json
import math
import os

import torch._inductor

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"

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

class NumericalStabilityProcessor(LogitsProcessor):
    
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-100.0, posinf=100.0, neginf=-100.0)
        return torch.clamp(scores, min=-100.0, max=100.0)

def calculate_perplexity_with_gpt_neo(text, model_name="./local_models/gpt-neo-125M"):

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(device)
        
        labels = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))

        perplexity = math.exp(loss.item())
        return perplexity

    except Exception as e:
        print(f"GPT-Neo perplexity calculation error: {e}")
        return None

OPENAI_GRADER_MODEL = os.getenv("OPENAI_GRADER_MODEL", "gpt-4o")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError, json.JSONDecodeError, ValueError))
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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,))
)
def llm_grade_explanation(explanation, insight, model_type="hybrid"):

    top_features = list(insight['top_features_impact'].keys())
    feature_impacts = []
    for feature, impact in insight['top_features_impact'].items():
        direction = "increases default risk" if impact > 0 else "decreases default risk"
        feature_impacts.append(f"{feature}: {direction} (importance: {impact:.3f})")
        
    predicted_prob = insight.get('predicted_proba', 0.5)
    approval_prob = 1 - predicted_prob
    
    model_context = """For HYBRID models: The explanation should demonstrate synthesis of individual borrower characteristics 
AND network/relational context. Credit for showing how personal factors interact with area/peer patterns.
Hybrid explanations are expected to be more comprehensive than single-modality explanations."""

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
- For hybrid models, both individual and network factors should be well-represented

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
        
class GNNConfig:
    
    IN_CHANNELS = 12
    HIDDEN_CHANNELS = 128
    OUT_CHANNELS = 1
    HEADS = 8
    EDGE_DIM = 4
    NODE_FEATURE_NAMES = [
        'fico', 'if_fthb', 'cnt_borr', 'cnt_units', 'dti', 'ltv', 'orig_upb',
        'loan_term', 'if_prim_res', 'if_corr', 'if_sf', 'if_purc'
    ]
    EDGE_FEATURE_NAMES = [
        'edge_type_area_connection',
        'edge_type_provider_connection',
        'edge_type_area_provider_connection',
        'inverse_group_size'
    ]

class GAT(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads,
                 attn_dropout=0.6, feat_dropout=0.5, edge_dim=None):
        
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels,
                                     heads=heads, concat=True,
                                     dropout=attn_dropout,
                                     edge_dim=edge_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_channels * heads)
        self.dropout = torch.nn.Dropout(feat_dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels,
                                     heads=1, concat=False,
                                     dropout=attn_dropout,
                                     edge_dim=edge_dim)
        
    def forward(self, x, edge_index, edge_attr):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = torch.nn.functional.elu(x)
        x = self.dropout(x)
        logits = self.conv2(x, edge_index, edge_attr)
        return torch.sigmoid(logits)

class GemmaHybridConfig:
    
    GEMMA_MODEL_PATH = "<my-path>/gemma-3-4b-it-finetuned-hybrid-8bit"
    BASE_MODEL_PATH = "<my-path>/gemma-3-4b-it"
    TRAINED_GAT_MODEL_PATH = "./trained_gat_model.pt"
    JULY_GRAPH_PATH = "../data/graph_data/july_explanation_graph.pt"
    TRAINED_XGB_MODEL_PATH = "./trained_xgb_model.pkl"
    XGB_TEST_DATA_PATH = "../data/cleaned_data/df_origination_test_scaled.csv"
    XGB_TRAINED_FEATURES = [
        'dt_orig', 'fico', 'mi_pct', 'cnt_units', 'dti', 'ltv', 'cnt_borr', 
        'orig_upb', 'loan_term', 'if_fthb', 'if_prim_res', 'if_corr', 'if_sf', 'if_purc'
    ]
    GAT_OPTIMAL_THRESHOLD = 0.45
    XGB_OPTIMAL_THRESHOLD = 0.4
    HYBRID_WEIGHT_GAT = 0.6
    HYBRID_WEIGHT_XGB = 0.4
    HYBRID_OPTIMAL_THRESHOLD = 0.5

class GemmaHybridPipeline:
    
    def __init__(self, config=None):
        
        self.config = config or GemmaHybridConfig()
        self.gat_model = None
        self.xgb_model = None
        self.graph_data = None
        self.X_test = None
        self.y_test = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gemma_model = None
        self.gemma_tokenizer = None
        self.shap_explainer = None
        self.gnn_explainer = None

    def load_models(self):
        
        print("Loading models...")

        self.gat_model = GAT(
            GNNConfig.IN_CHANNELS,
            GNNConfig.HIDDEN_CHANNELS,
            GNNConfig.OUT_CHANNELS,
            GNNConfig.HEADS,
            edge_dim=GNNConfig.EDGE_DIM
        )
        self.gat_model.load_state_dict(torch.load(self.config.TRAINED_GAT_MODEL_PATH, map_location=self.device))
        self.gat_model.eval()
        self.gat_model.to(self.device)
        print("GAT model loaded.")

        self.xgb_model = joblib.load(self.config.TRAINED_XGB_MODEL_PATH)
        print("XGBoost model loaded.")
        return True

    def load_data(self):
        
        print("Loading data...")
        if not os.path.exists(self.config.JULY_GRAPH_PATH):
            print(f"Graph data not found at '{self.config.JULY_GRAPH_PATH}'")
            return False
        with open(self.config.JULY_GRAPH_PATH, 'rb') as f:
            self.graph_data = torch.load(f, map_location=self.device, weights_only=False)
        print("Graph data loaded.")
        if hasattr(self.graph_data, 'edge_index') and self.graph_data.edge_index is not None:
            self.graph_data.edge_index = self.graph_data.edge_index.long().contiguous()
        if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
            self.graph_data.edge_attr = self.graph_data.edge_attr.float().contiguous()
        if hasattr(self.graph_data, 'default') and self.graph_data.default is not None:
            self.graph_data.y = self.graph_data.default.float()
            del self.graph_data.default
        elif hasattr(self.graph_data, 'y') and self.graph_data.y is not None:
            self.graph_data.y = self.graph_data.y.float()
        if not os.path.exists(self.config.XGB_TEST_DATA_PATH):
            print(f"Tabular data not found at '{self.config.XGB_TEST_DATA_PATH}'")
            return False
        test_data = pd.read_csv(self.config.XGB_TEST_DATA_PATH)
        exclude_cols = ['id', 'id_loan', 'year', 'month', 'provider', 'default']
        self.X_test = test_data.copy()
        self.X_test.drop(columns=[c for c in exclude_cols if c in self.X_test.columns and c != 'id_loan'], inplace=True)
        if 'default' in test_data.columns:
            self.y_test = test_data['default']
        else:
            self.y_test = None
        cols_to_check = [col for col in self.X_test.columns if col != 'id_loan']
        non_numeric_cols = [col for col in cols_to_check if self.X_test[col].dtype == 'O']
        if non_numeric_cols:
            print(f"Converting non-numeric columns to numeric: {non_numeric_cols}")
            self.X_test[non_numeric_cols] = self.X_test[non_numeric_cols].apply(pd.to_numeric, errors='coerce')
        if self.X_test[cols_to_check].isnull().any().any():
            print("Filling NaNs in X_test (excluding id_loan) with 0.")
            self.X_test[cols_to_check] = self.X_test[cols_to_check].fillna(0)
        print("Tabular data loaded and cleaned.")
        return True
    
    def initialize_gemma(self):

        print("Loading fine-tuned Gemma-3 model...")
        
        try:
            print(f"Loading base model from: {self.config.BASE_MODEL_PATH}")
            
            precision = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.BASE_MODEL_PATH,
                torch_dtype=precision,
                device_map="cuda:0",
                trust_remote_code=True
            )
            
            print("Loading tokenizer...")
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(self.config.BASE_MODEL_PATH)
            
            if self.gemma_tokenizer.pad_token is None:
                self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
            
            print(f"Loading LoRA adapter from: {self.config.GEMMA_MODEL_PATH}")
            self.gemma_model = PeftModel.from_pretrained(
                base_model, 
                self.config.GEMMA_MODEL_PATH,
                torch_dtype=torch.float16
            )
            
            self.gemma_model = self.gemma_model.merge_and_unload()
            self.gemma_model.eval()
            
            print("Gemma-3 model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading Gemma-3 model: {e}")
            return False

    def initialize_explainers(self):
        
        print("Initializing explainers...")
        try:
                self.shap_explainer = shap.TreeExplainer(self.xgb_model, background_sample)
                print("SHAP explainer initialized")
            if self.gat_model is not None:
                self.gnn_explainer = Explainer(
                    model=self.gat_model,
                    algorithm=GNNExplainer(epochs=200),
                    explanation_type='model',
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='binary_classification',
                        task_level='node',
                        return_type='probs'
                    )
                )
                print("GNNExplainer initialized")
            return True
        except Exception as e:
            print(f"Error initializing explainers: {e}")
            return False

    def get_shap_explanation(self, node_idx, top_k=5):
        
        print(f"Generating SHAP explanation for node {node_idx}...")
        if self.shap_explainer is None:
            print("SHAP explainer not available")
            return {}, "SHAP explainer not available"
        try:
            xgb_input = self._prepare_xgb_input(node_idx)
            shap_values = self.shap_explainer.shap_values(xgb_input)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            feature_order = self.config.XGB_TRAINED_FEATURES
            if hasattr(self.xgb_model, 'feature_names_in_'):
                feature_order = list(self.xgb_model.feature_names_in_)
            feature_importance = {}
            for i, feature in enumerate(feature_order):
                if feature in xgb_input.columns:
                    shap_val = shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]
                    feature_importance[feature] = {
                        'shap_value': float(shap_val),
                        'feature_value': float(xgb_input[feature].iloc[0]),
                        'contribution': 'increases' if shap_val > 0 else 'decreases'
                    }
            top_features = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]['shap_value']), reverse=True)[:top_k])
            shap_summary = []
            for feature, info in top_features.items():
                shap_summary.append(
                    f"{feature} (value: {info['feature_value']:.3f}, SHAP: {info['shap_value']:.3f}, {info['contribution']} risk)"
                )
            return top_features, "; ".join(shap_summary)
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return {}, f"Error generating SHAP explanation: {e}"

    def get_gnn_explanation(self, node_idx):
        
        print(f"Generating GNN explanation for node {node_idx}...")

        try:
            explanation = self.gnn_explainer(
                x=self.graph_data.x,
                edge_index=self.graph_data.edge_index,
                edge_attr=self.graph_data.edge_attr,
                index=node_idx
            )
            edge_mask = explanation.edge_mask
            node_mask = explanation.node_mask if hasattr(explanation, 'node_mask') else None
            if edge_mask is not None:
                edge_importance = edge_mask.detach().cpu().numpy()
                top_edge_indices = np.argsort(edge_importance)[-5:]
                network_summary = []
                for edge_idx in top_edge_indices:
                    if edge_idx < len(self.graph_data.edge_index[0]):
                        source = int(self.graph_data.edge_index[0][edge_idx])
                        target = int(self.graph_data.edge_index[1][edge_idx])
                        importance = float(edge_importance[edge_idx])
                        edge_type = "connection"
                        if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
                            edge_features = self.graph_data.edge_attr[edge_idx].detach().cpu().numpy()
                            if len(edge_features) >= 4:
                                if edge_features[0] > 0.5:
                                    edge_type = "area_connection"
                                elif edge_features[1] > 0.5:
                                    edge_type = "provider_connection"
                                elif edge_features[2] > 0.5:
                                    edge_type = "area_provider_connection"
                        network_summary.append(
                            f"{edge_type} between nodes {source}-{target} (importance: {importance:.3f})"
                        )
                network_text = "; ".join(network_summary)
            else:
                network_text = "Edge importance scores not available"
            return {
                'edge_mask': edge_mask,
                'node_mask': node_mask,
                'important_edges': top_edge_indices.tolist() if 'top_edge_indices' in locals() else [],
                'network_summary': network_text
            }, network_text
        except Exception as e:
            print(f"GNN explanation error: {e}")
            return {}, f"Error generating GNN explanation: {e}"

    def get_hybrid_prediction(self, node_idx):

        print(f"Generating hybrid prediction for node {node_idx}...")

        try:
            with torch.no_grad():
                gat_proba = self.gat_model(
                    self.graph_data.x,
                    self.graph_data.edge_index,
                    self.graph_data.edge_attr
                )[node_idx].squeeze().item()
        except Exception as e:
            print(f"GAT prediction error: {e}")
            gat_proba = 0.5

        xgb_input = self._prepare_xgb_input(node_idx)
        if xgb_input is not None:
            try:
                xgb_input_array = xgb_input[self.config.XGB_TRAINED_FEATURES].values
                xgb_proba = self.xgb_model.predict_proba(xgb_input_array)[:, 1][0]
            except Exception as e:
                print(f"XGBoost prediction error: {e}")
                try:
                    xgb_proba = self.xgb_model.predict_proba(xgb_input)[:, 1][0]
                except Exception as e2:
                    print(f"XGBoost fallback also failed: {e2}")
                    xgb_proba = 0.5
        else:
            xgb_proba = 0.5

        hybrid_proba = (self.config.HYBRID_WEIGHT_GAT * gat_proba + 
                        self.config.HYBRID_WEIGHT_XGB * xgb_proba)

        feature_values = {}
        if xgb_input is not None:
            for feature in self.config.XGB_TRAINED_FEATURES:
                if feature in xgb_input.columns:
                    feature_values[feature] = xgb_input[feature].iloc[0]

        return {
            'gat_proba': gat_proba,
            'xgb_proba': xgb_proba,
            'hybrid_proba': hybrid_proba,
            'gat_prediction': 'Default' if gat_proba >= self.config.GAT_OPTIMAL_THRESHOLD else 'No Default',
            'xgb_prediction': 'Default' if xgb_proba >= self.config.XGB_OPTIMAL_THRESHOLD else 'No Default',
            'hybrid_prediction': 'Default' if hybrid_proba >= self.config.HYBRID_OPTIMAL_THRESHOLD else 'No Default',
            'feature_values': feature_values
        }

    def _prepare_xgb_input(self, node_idx):

        single_instance = self.X_test.iloc[[node_idx]]
        available_features = [feat for feat in self.config.XGB_TRAINED_FEATURES 
                              if feat in single_instance.columns]
        return single_instance[self.config.XGB_TRAINED_FEATURES]

    def build_enhanced_gemma_prompt(self, predictions, shap_info, gnn_info):
        
        hybrid_proba = predictions['hybrid_proba']
        hybrid_pred = predictions['hybrid_prediction']
        gat_pred = predictions['gat_prediction']
        xgb_pred = predictions['xgb_prediction']
        top_shap_features, shap_summary = shap_info
        gnn_explanation, network_summary = gnn_info
        current_node_idx = getattr(self, '_current_node_idx', 0)

        if hasattr(self, 'X_test') and current_node_idx < len(self.X_test):
            df_row = self.X_test.iloc[current_node_idx]

        if df_row is not None:
            customer_factors = []
            improvement_advice = []
            top_factors_formatted = []
            factor_count = 1

            for feature, feature_data in list(top_shap_features.items())[:6]:

                if isinstance(feature_data, dict):
                    shap_value = feature_data.get('shap_value', 0)
                    feature_value = feature_data.get('feature_value', 0)
                else:
                    shap_value = feature_data
                    feature_value = df_row.get(feature, 0) if hasattr(df_row, 'get') else 0

                if feature == 'fico':
                    credit_explanation = self._get_credit_score_explanation(df_row)
                    customer_factors.append(credit_explanation)
                    top_factors_formatted.append(f"{factor_count}. {credit_explanation}")
                else:
                    plain_name, pop_context, impact_level, direction, advice = self._translate_features_to_customer_language(
                        feature, feature_value, shap_value
                    )

                    factor_text = f"Your {plain_name} {pop_context}"
                    customer_factors.append(factor_text)
                    top_factors_formatted.append(f"{factor_count}. {factor_text}")

                    if "hurts" in direction and impact_level in ["really", "significantly"]:
                        improvement_advice.append(advice)

                factor_count += 1
        else:
            credit_explanation = "Your credit score affects your approval chances"
            customer_factors = [credit_explanation, "your financial situation"]
            improvement_advice = ["keep managing your finances well"]
            top_factors_formatted = ["1. Your credit history", "2. Your financial situation"]

        if hasattr(gnn_explanation, 'get'):
            gnn_data = gnn_explanation
        else:
            gnn_data = {
                'top_edges_impact': [],
                'node_index': current_node_idx
            }

        network_context = self._translate_network_to_customer_language(gnn_data)

        if len(customer_factors) <= 2:
            factors_text = " and ".join(customer_factors)
        elif len(customer_factors) == 3:
            factors_text = f"{customer_factors[0]}, {customer_factors[1]}, and {customer_factors[2]}"
        else:
            factors_text = f"{', '.join(customer_factors[:3])}, and {customer_factors[3]}"

        advice_text = improvement_advice[0] if improvement_advice else "keep managing your finances well"

        approval_chance = (1 - hybrid_proba) * 100
        if approval_chance > 85:
            approval_context = f"excellent approval chances ({approval_chance:.0f}% likely to get approved)"
        elif approval_chance > 65:
            approval_context = f"strong approval chances ({approval_chance:.0f}% likely to get approved)"
        elif approval_chance > 50:
            approval_context = f"reasonable approval chances ({approval_chance:.0f}% likely to get approved)"
        else:
            approval_context = f"challenging approval process ({approval_chance:.0f}% likely to get approved)"

        if gat_pred != xgb_pred:
            conflict_resolution = f"When we look at your personal finances, we see one picture, but when we consider your area and similar borrowers, we see another. Our final assessment balances both perspectives to give you the most accurate evaluation."
        else:
            conflict_resolution = f"Both your personal financial profile and your area patterns point to the same result, which gives us confidence in this assessment."

        system_prompt = GLOBAL_SYSTEM_PROMPT

        user_prompt = f"""
    You are explaining to someone why their loan application has {approval_context} after looking at everything - their personal finances, their local area, and similar borrowers.

    COMPLETE PICTURE:
    Personal finances: {factors_text}
    Local context: {network_context}

    DETAILED BREAKDOWN (mention ALL of these factors):
    {chr(10).join(top_factors_formatted)}

    Area factors: {network_context}

    WRITE AN EXPLANATION that a smart 17-year-old would understand. Follow these rules:

    1. Start with what their {approval_chance:.0f}% approval chance means overall
    2. Always mention their credit score situation first
    3. Discuss AT LEAST {min(len(top_factors_formatted), 5)} personal financial factors from the detailed breakdown above
    4. Explain how their area/location adds context with SPECIFIC NUMBERS
    5. Show how personal + area factors combine to create their final result
    6. Give specific advice: {advice_text}
    7. Mention how they could improve both personal and situational factors
    8. End with encouraging perspective about their complete picture

    CRITICAL REQUIREMENTS:
    - Never use banking jargon, model names, or technical terms
    - Always include SPECIFIC NUMBERS when mentioning connections (e.g., "3 other borrowers," not "several folks")
    - Cover the COMPLETE picture - don't skip important factors from the breakdown
    - Compare to "other people your age," "people in your area," "typical borrowers"
    - Explain WHY each factor matters in real life
    - Give concrete timelines and improvement amounts
    - Sound like a helpful mentor teaching about money and location
    - Keep to 8-10 sentences total but ensure comprehensive coverage

    CONTEXT NOTE: {conflict_resolution}

    TONE: Encouraging teacher explaining how personal finance and local economics work together.
    """

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

        return messages
    
    def get_gemma_explanation_enhanced(self, predictions, shap_info, gnn_info):

        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True

            messages = self.build_enhanced_gemma_prompt(predictions, shap_info, gnn_info)

            formatted_prompt = self.gemma_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.gemma_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4000,
                padding=True
            )

            device = next(self.gemma_model.parameters()).device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            print("Generating explanation with Gemma-3...")

            self.gemma_model.eval()
            torch.cuda.empty_cache()
            
            with torch.inference_mode():
                outputs = self.gemma_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.gemma_tokenizer.pad_token_id,
                    eos_token_id=self.gemma_tokenizer.eos_token_id,
                    logits_processor=[NumericalStabilityProcessor()]
                )

            generated_tokens = outputs[:, input_ids.shape[1]:] 
            explanation = self.gemma_tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            explanation = explanation.strip()

            stop_patterns = [
                "user\n",
                "GUIDED SUBGOAL STRUCTURE",
                "OUTPUT LENGTH/FORMAT CONSTRAINT",
                "COMBINED BORROWER-LEVEL",
                "COMBINED INSTRUCTIONAL",
                "CHAIN-OF-THOUGHT",
                "DISAMBIGUATION HINT",
                "\n\nmodel",
                "\nmodel",
                "\n\n"
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
        
        
    def _get_credit_score_explanation(self, df_row):

        if hasattr(df_row, 'get'):
            if 'fico_actual' in df_row.index and not pd.isna(df_row['fico_actual']):
                actual_score = int(df_row['fico_actual'])
            elif 'fico' in df_row.index:
                normalized_fico = df_row['fico']
                actual_score = int(300 + (normalized_fico * 550))
            else:
                return "Your credit score is about average compared to other borrowers"
        else:
            return "Your credit score affects your approval chances"

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

    def _translate_features_to_customer_language(self, feature, feature_value, shap_value):

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

    def _translate_network_to_customer_language(self, gnn_data):

        top_edges = gnn_data.get('top_edges_impact', [])
        current_node = gnn_data.get('node_index', 0)

        if not top_edges:
            if hasattr(self, 'graph_data') and hasattr(self.graph_data, 'edge_index'):
                edge_index = self.graph_data.edge_index
                node_edges = ((edge_index[0] == current_node) | (edge_index[1] == current_node)).sum().item()

                if node_edges > 0:
                    return f"you have {node_edges} connections to other borrowers in our database (showing neutral to slightly positive patterns for approval)"
                else:
                    return "you don't have significant connections to other borrowers in our analysis"
            else:
                return "you don't have significant connections to other borrowers in our analysis"

        area_borrowers = set()
        provider_borrowers = set()  
        area_provider_borrowers = set()
        area_risks = []
        provider_risks = []
        area_provider_risks = []

        for edge in top_edges:
            attrs = edge.get('attributes', {})
            importance = edge.get('importance', 0)
            src = edge.get('source', 0)
            dst = edge.get('destination', 0)
            other_borrower = dst if src == current_node else src

            if attrs.get('edge_type_area_connection', 0) > 0.5:
                area_borrowers.add(other_borrower)
                area_risks.append(importance)
            elif attrs.get('edge_type_provider_connection', 0) > 0.5:
                provider_borrowers.add(other_borrower)
                provider_risks.append(importance)
            elif attrs.get('edge_type_area_provider_connection', 0) > 0.5:
                area_provider_borrowers.add(other_borrower)
                area_provider_risks.append(importance)

        connection_parts = []

        if area_borrowers:
            avg_area_risk = sum(area_risks) / len(area_risks) if area_risks else 0
            risk_descriptor = "concerning" if avg_area_risk < -0.02 else "positive" if avg_area_risk > 0.02 else "neutral"
            borrower_count = len(area_borrowers)
            connection_parts.append(f"you're connected to {borrower_count} other borrower{'s' if borrower_count != 1 else ''} in your geographic area (showing {risk_descriptor} patterns)")

        if provider_borrowers:
            avg_provider_risk = sum(provider_risks) / len(provider_risks) if provider_risks else 0
            risk_descriptor = "concerning" if avg_provider_risk < -0.02 else "positive" if avg_provider_risk > 0.02 else "neutral"
            borrower_count = len(provider_borrowers)
            connection_parts.append(f"you share the same lender type with {borrower_count} other borrower{'s' if borrower_count != 1 else ''} (showing {risk_descriptor} patterns)")

        if area_provider_borrowers:
            avg_combined_risk = sum(area_provider_risks) / len(area_provider_risks) if area_provider_risks else 0
            risk_descriptor = "concerning" if avg_combined_risk < -0.02 else "positive" if avg_combined_risk > 0.02 else "neutral"
            borrower_count = len(area_provider_borrowers)
            connection_parts.append(f"you're connected to {borrower_count} other borrower{'s' if borrower_count != 1 else ''} who both live in your area AND use the same lender type (showing {risk_descriptor} patterns)")

        if not connection_parts:
            unique_borrowers = set()
            for edge in top_edges:
                src = edge.get('source', 0)
                dst = edge.get('destination', 0)
                other_borrower = dst if src == current_node else src
                unique_borrowers.add(other_borrower)

            if unique_borrowers:
                total_count = len(unique_borrowers)
                return f"you have connections to {total_count} other borrower{'s' if total_count != 1 else ''} through mixed geographic and lender relationships (showing neutral patterns for your approval)"
            else:
                return "you have minimal connections to other borrowers in our analysis (which is actually neutral for your application)"

        all_unique_borrowers = area_borrowers | provider_borrowers | area_provider_borrowers
        total_borrowers = len(all_unique_borrowers)
        all_importances = [edge.get('importance', 0) for edge in top_edges]
        avg_importance = sum(all_importances) / len(all_importances) if all_importances else 0

        if avg_importance > 0.05:
            overall_impact = f"These connections to {total_borrowers} total borrower{'s' if total_borrowers != 1 else ''} help your approval chances"
        elif avg_importance < -0.05:
            overall_impact = f"These connections to {total_borrowers} total borrower{'s' if total_borrowers != 1 else ''} create some concern for your approval"
        else:
            overall_impact = f"These connections to {total_borrowers} total borrower{'s' if total_borrowers != 1 else ''} have a neutral effect on your application"

        if len(connection_parts) == 1:
            return f"{connection_parts[0]}. {overall_impact}."
        elif len(connection_parts) == 2:
            return f"{connection_parts[0]} and {connection_parts[1]}. {overall_impact}."
        else:
            return f"{', '.join(connection_parts[:-1])}, and {connection_parts[-1]}. {overall_impact}."
        
    def get_comprehensive_explanation(self, node_idx):
        
        print(f"\n###\nStarting comprehensive explanation for node {node_idx}...")

        self._current_node_idx = node_idx

        try:
            predictions = self.get_hybrid_prediction(node_idx)
            shap_features, shap_summary = self.get_shap_explanation(node_idx, top_k=5)
            gnn_explanation, network_summary = self.get_gnn_explanation(node_idx)

            if isinstance(gnn_explanation, dict):
                gnn_data_for_translation = {
                    'node_index': node_idx,
                    'top_edges_impact': []
                }

                if 'important_edges' in gnn_explanation and hasattr(self.graph_data, 'edge_index'):
                    for edge_idx in gnn_explanation.get('important_edges', []):
                        if edge_idx < len(self.graph_data.edge_index[0]):
                            source = int(self.graph_data.edge_index[0][edge_idx])
                            target = int(self.graph_data.edge_index[1][edge_idx])
                            importance = 0.1

                            edge_attrs = {}
                            if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
                                if edge_idx < len(self.graph_data.edge_attr):
                                    edge_features = self.graph_data.edge_attr[edge_idx].detach().cpu().numpy()
                                    if len(edge_features) >= 4:
                                        edge_attrs = {
                                            'edge_type_area_connection': float(edge_features[0]),
                                            'edge_type_provider_connection': float(edge_features[1]),
                                            'edge_type_area_provider_connection': float(edge_features[2]),
                                            'inverse_group_size': float(edge_features[3])
                                        }

                            gnn_data_for_translation['top_edges_impact'].append({
                                'source': source,
                                'destination': target,
                                'importance': importance,
                                'attributes': edge_attrs
                            })
            else:
                gnn_data_for_translation = {'node_index': node_idx, 'top_edges_impact': []}

            gemma_explanation = self.get_gemma_explanation_enhanced(
                predictions, (shap_features, shap_summary), (gnn_data_for_translation, network_summary)
            )

            return gemma_explanation.strip()

        except Exception as e:
            print(f"Error generating comprehensive explanation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating comprehensive explanation: {e}"

    def _format_shap_features(self, shap_features):

        formatted = []
        for feature, info in shap_features.items():
            formatted.append(
                f"- {feature}: {info['feature_value']:.3f} (SHAP: {info['shap_value']:.3f}, {info['contribution']} default risk)"
            )
        return "\n".join(formatted)

    def validate_loan_alignment(self, node_idx):

        if hasattr(self.graph_data, 'loan_id') and 'id_loan' in self.X_test.columns:
            graph_loan_id = self.graph_data.loan_id[node_idx]
            tabular_loan_id = self.X_test.iloc[node_idx]['id_loan']
            return int(graph_loan_id) == int(tabular_loan_id)
        return True

    def run_hybrid_demo(self, node_idx=None, num_samples_perplexity=None, selected_indices=None):
    
        print("Bimodal Explanation Pipeline (Fixed)")

        os.makedirs("explanations", exist_ok=True)
        explanation_file = "explanations/gemma_hybrid_explanations.jsonl"

        max_nodes = min(self.graph_data.num_nodes, len(self.X_test))

        if selected_indices is not None:
            valid_indices = []
            for idx in selected_indices:
                if idx < max_nodes and self.validate_loan_alignment(idx):
                    valid_indices.append(idx)

                selected_indices = None

        if selected_indices is None:
            if node_idx is None:
                print(f"\n### Selecting {num_samples_perplexity} different random borrowers ###")
                visited_nodes = set()
                node_indices = []

                for i in range(num_samples_perplexity):
                    attempts = 0
                    max_attempts = min(50, max_nodes) 

                    while attempts < max_attempts:
                        candidate_idx = random.randint(0, max_nodes - 1)

                        if candidate_idx in visited_nodes:
                            attempts += 1
                            continue

                        if self.validate_loan_alignment(candidate_idx):
                            visited_nodes.add(candidate_idx)
                            node_indices.append(candidate_idx)
                            print(f"Selected node {candidate_idx} for sample {i+1}")
                            break

        print(f"\n### Evaluation with {len(node_indices)} samples ###")

        perplexities = []
        llm_scores = {
            "Individual Feature Coverage": [], 
            "Individual Feature Consistency": [],
            "Network Coverage": [],
            "Network Consistency": []
        }
        explanations = []
        node_results = []

        for i, current_node_idx in enumerate(node_indices):
            print(f"\n### Sample {i+1}/{len(node_indices)} - Analyzing Node {current_node_idx} ###")

            try:
                print(f"Generating explanation for node {current_node_idx}...")
                explanation = self.get_comprehensive_explanation(current_node_idx)
                explanations.append(explanation)
                node_results.append(current_node_idx)

                print(f"Calculating perplexity for node {current_node_idx}...")
                perplexity = calculate_perplexity_with_gpt_neo(explanation)
                if perplexity is not None:
                    perplexities.append(perplexity)
                    print(f"Node {current_node_idx} Perplexity: {perplexity:.2f}")

                print(f"Grading explanation for node {current_node_idx}...")
                predictions = self.get_hybrid_prediction(current_node_idx)
                shap_features, _ = self.get_shap_explanation(current_node_idx, top_k=5)

                insight_for_grading = {
                    'predicted_proba': predictions['hybrid_proba'],
                    'top_features_impact': {
                        feature: info['shap_value']
                        for feature, info in shap_features.items()
                    }
                }

                grades = llm_grade_explanation(explanation, insight_for_grading, model_type="hybrid")
                if grades:
                    print(f"Node {current_node_idx} Grades: {grades}")
                    for metric, score in grades.items():
                        llm_scores[metric].append(score)

            except Exception as e:
                print(f"Error analyzing node {current_node_idx}: {e}")
                continue

        print(f"\n### Final Summary ###")
        print(f"Analyzed nodes: {node_results}")
        print(f"Generated {len(explanations)} explanations")

        if explanations:
            print(f"Example explanation (Node {node_results[0]}):\n{explanations[0]}")

        print("\n### 95% Confidence Intervals ###")
        for name, data in [
            ("Perplexity", perplexities),
            ("Individual Feature Coverage", llm_scores["Individual Feature Coverage"]),
            ("Individual Feature Consistency", llm_scores["Individual Feature Consistency"]),
            ("Network Coverage", llm_scores["Network Coverage"]),
            ("Network Consistency", llm_scores["Network Consistency"])
        ]:
            mean_score, ci_str = compute_confidence_interval(data)
            if mean_score is None:
                print(f"{name}: No data available")
                continue
            if name == "Perplexity" and len(data) > 1 and np.std(data) > 0:
                h = stats.sem(data) * stats.t.ppf(0.975, len(data) - 1)
                ci_low_bounded = max(1.0, mean_score - h)
                ci_high = mean_score + h
                h_bounded = (ci_high - ci_low_bounded) / 2
                if mean_score - h < 1.0:
                    print(f"{name}: {mean_score:.2f} ± {h_bounded:.2f} (95% CI = [{ci_low_bounded:.2f}, {ci_high:.2f}]) *bounded")
                else:
                    print(f"{name}: {ci_str}")
            elif name in ["Individual Feature Coverage", "Individual Feature Consistency", 
             "Network Coverage", "Network Consistency"] and len(data) > 1 and np.std(data) > 0:
                h = stats.sem(data) * stats.t.ppf(0.975, len(data) - 1)
                ci_low_bounded = max(1.0, mean_score - h)
                ci_high_bounded = min(5.0, mean_score + h)
                h_bounded = (ci_high_bounded - ci_low_bounded) / 2
                bounds_applied = (mean_score - h < 1.0) or (mean_score + h > 5.0)
                if bounds_applied:
                    print(f"{name}: {mean_score:.2f} ± {h_bounded:.2f} (95% CI = [{ci_low_bounded:.2f}, {ci_high_bounded:.2f}]) *bounded")
                else:
                    print(f"{name}: {ci_str}")
            else:
                print(f"{name}: {ci_str}")

        with open(explanation_file, 'a', encoding='utf-8') as f:
            for node_idx, explanation in zip(node_results, explanations):
                record = {
                    'node_index': node_idx,
                    'explanation': explanation
                }

                f.write(json.dumps(record) + '\n')

        print(f"Saved {len(explanations)} explanations to {explanation_file}")

        return {
            'explanations': explanations,
            'perplexities': perplexities,
            'llm_scores': llm_scores,
            'node_indices': node_results  
        }

if __name__ == "__main__":

    pipeline = GemmaHybridPipeline()
    selected_indices = None
    indices_file = "../evaluations/gemini_xgb/selected_borrower_indices.json"

    with open(indices_file, 'r') as f:
        selected_indices = json.load(f)
    print(f"Loaded {len(selected_indices)} selected indices from {indices_file}")
    
    pipeline.run_hybrid_demo(num_samples_perplexity=100, selected_indices=selected_indices)