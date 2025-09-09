"""
===============================================================================
DeepSeek R1 70B Fine-tuned GAT Loan Default Explanation and Evaluation Pipeline
===============================================================================

This file implements a comprehensive evaluation framework for loan default predictions using GATs with fine-tuned DeepSeek R1 70B language model explanations focused on network 
relationships and structural patterns. The system generates network-aware explanations that prioritize graph connectivity patterns, geographic clustering, and lender relationship effects while evaluating explanation quality through automated scoring metrics.

The pipeline emphasizes network relationships including geographic area connections, loan provider relationships, and area-provider clustering effects, integrating these structural insights with individual borrower characteristics to provide comprehensive risk assessments.

Note: This script is intended for academic reference only.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.algorithm import GNNExplainer as GNNExplainerAlgorithm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from transformers import LogitsProcessor

from openai import OpenAI
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from dotenv import load_dotenv

import random
import json
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
""".strip()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_GRADER_MODEL = 'gpt-4o'

class NumericalStabilityProcessor(LogitsProcessor):
    
    def __call__(self, input_ids, scores):
        
        scores = torch.nan_to_num(scores, nan=-100.0, posinf=100.0, neginf=-100.0)
        return torch.clamp(scores, min=-100.0, max=100.0)

class DeepSeekGATConfig:
    
    DEEPSEEK_MODEL_PATH = "<my-path>/deepseek-r1-70b-it-finetuned-gat-8bit"
    BASE_MODEL_PATH = "<my-path>/DeepSeek-R1-Distill-Llama-70B"
    
class Config:

    TRAINED_MODEL_PATH = "./trained_gat_model.pt"
    JULY_GRAPH_PATH = "../data/graph_data/july_explanation_graph.pt"
    IN_CHANNELS = 12
    HIDDEN_CHANNELS = 128
    OUT_CHANNELS = 1
    HEADS = 8
    EDGE_DIM = 4
    EXPLAINER_EPOCHS = 100
    EXPLAINER_LR = 0.01
    NUM_TOP_EXPLAIN_FEATURES = 5
    NUM_TOP_EXPLAIN_EDGES = 5
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
    NUM_SAMPLES = 100
    EXPLANATIONS_SAVE_DIR = "explanations"
    EXPLANATIONS_FILE_NAME = "deepseek_gnn_explanations.jsonl"

class GAT(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads,
                 attn_dropout=0.6, feat_dropout=0.5, edge_dim=None):
        
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels,
                                     heads=heads, concat=True,
                                     dropout=attn_dropout,
                                     edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.dropout = nn.Dropout(feat_dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels,
                                     heads=1, concat=False,
                                     dropout=attn_dropout,
                                     edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        logits = self.conv2(x, edge_index, edge_attr)
        return torch.sigmoid(logits)
    
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

def calculate_perplexity_with_gpt_neo(text, model_name="./local_models/gpt-neo-125M"):

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import math

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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError, json.JSONDecodeError, ValueError))
)
def llm_grade_explanation(explanation, insight, model_type="gnn"):
    
    top_features = list(insight['top_features_impact'].keys())
    feature_impacts = []
    for feature, impact in insight['top_features_impact'].items():
        direction = "increases default risk" if impact > 0 else "decreases default risk"
        feature_impacts.append(f"{feature}: {direction} (importance: {impact:.3f})")
        
    predicted_prob = insight.get('predicted_proba', 0.5)
    approval_prob = 1 - predicted_prob
    
    model_context = """For GNN models: The explanation should demonstrate synthesis of individual borrower characteristics 
AND network/relational context. GNN models explicitly capture network relationships and node connections.
Both individual feature coverage and network coverage should be substantial for high-quality explanations."""

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
Evaluates discussion of network/relational factors and connections. For GNN models, expect substantial network content.
- 5: Substantial network discussion with specific details (borrower counts, connection types, area patterns)
- 4: Good network focus with adequate detail about connections or area factors
- 3: Moderate network content - mentions connections but lacks depth or specificity
- 2: Limited network discussion - brief mentions without meaningful detail
- 1: No meaningful network or relational content

**Network Consistency (1-5):**
Evaluates accuracy of network relationship descriptions and risk implications. For GNN models, this should be detailed.
- 5: Accurate and logical network relationship descriptions with clear risk implications
- 4: Generally correct network interpretations with sound reasoning
- 3: Basic network understanding with mostly logical explanations
- 2: Some network logic but with unclear or inconsistent elements
- 1: Incorrect or contradictory network relationship descriptions

**IMPORTANT NOTES:**
- Customer-friendly language (e.g., "credit score" instead of "fico") is acceptable and preferred
- For GNN models, both individual and network metrics are important
- Network explanations should include specific details like borrower counts and connection types

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
        
def load_graph_data(path):
    
    data = torch.load(path)

    if hasattr(data, 'edge_index') and data.edge_index is not None:
        if data.edge_index.dtype != torch.long:
            print(f"Warning: Converting edge_index from {data.edge_index.dtype} to torch.long")
            data.edge_index = data.edge_index.long()
        if not data.edge_index.is_contiguous():
            print(f"Warning: Making edge_index contiguous")
            data.edge_index = data.edge_index.contiguous()
            
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        if data.edge_attr.dtype != torch.float:
            print(f"Warning: Converting edge_attr from {data.edge_attr.dtype} to torch.float")
            data.edge_attr = data.edge_attr.float()
        if not data.edge_attr.is_contiguous():
            print(f"Warning: Making edge_attr contiguous")
            data.edge_attr = data.edge_attr.contiguous()
    else:
        if Config.EDGE_DIM > 0:
            raise ValueError(f"Graph data is missing 'edge_attr' but Config.EDGE_DIM is set to {Config.EDGE_DIM}")

    if hasattr(data, 'default') and data.default is not None:
        labels = data.default
        if labels.dtype != torch.float:
            print(f"Warning: Converting 'default' labels from {labels.dtype} to torch.float")
            data.y = labels.float()
        else:
            data.y = labels
        del data.default
    elif hasattr(data, 'y') and data.y is not None:
        if data.y.dtype != torch.float:
            print(f"Warning: Converting existing 'y' labels from {data.y.dtype} to torch.float")
            data.y = data.y.float()
    else:
        print(f"Warning: No 'y' or 'default' labels found")

    return data

def load_trained_gat_model(model_path, in_channels, hidden_channels, out_channels, heads, edge_dim):

    model = GAT(in_channels, hidden_channels, out_channels, heads, edge_dim=edge_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Trained GAT model loaded successfully from {model_path}")
    return model

def get_gnnexplainer_insights_for_node(model, graph_data, node_idx_to_explain, num_top_features, num_top_edges):

    model_config = {
        "mode": "binary_classification",
        "task_level": "node",
        "return_type": "probs"
    }

    algorithm = GNNExplainerAlgorithm(
        epochs=Config.EXPLAINER_EPOCHS,
        lr=Config.EXPLAINER_LR
    )

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config
    )

    x_data = graph_data.x.to('cpu')
    edge_index_data = graph_data.edge_index.to('cpu')
    edge_attr_data = graph_data.edge_attr.to('cpu')

    print(f"Explaining node {node_idx_to_explain + 1}...")
    
    try:
        explanation = explainer(
            x=x_data,
            edge_index=edge_index_data,
            edge_attr=edge_attr_data,
            index=node_idx_to_explain
        )
    except Exception as e:
        print(f"Explanation failed for node {node_idx_to_explain}: {e}")
        return None

    feat_masks = explanation.node_mask
    node_feat_mask = feat_masks[node_idx_to_explain].abs()

    if node_feat_mask.sum() > 0:
        node_feat_mask = node_feat_mask / node_feat_mask.sum()
    
    mask_arr = node_feat_mask.cpu().numpy()

    if len(mask_arr) != len(Config.NODE_FEATURE_NAMES):
        print(f"Feature mask length ({len(mask_arr)}) doesn't match feature names ({len(Config.NODE_FEATURE_NAMES)})")
        min_len = min(len(mask_arr), len(Config.NODE_FEATURE_NAMES))
        mask_arr = mask_arr[:min_len]
        feature_names = Config.NODE_FEATURE_NAMES[:min_len]
    else:
        feature_names = Config.NODE_FEATURE_NAMES

    feature_impact = pd.DataFrame({
        'feature': feature_names,
        'importance': mask_arr
    }).sort_values(by='importance', key=abs, ascending=False)

    top_features_impact = {}
    for _, row in feature_impact.head(num_top_features).iterrows():
        top_features_impact[row['feature']] = float(row['importance'])

    top_edges_impact_list = []
    edge_mask = explanation.edge_mask
    
    if edge_mask is not None and edge_mask.numel() > 0:
        edge_mask_np = edge_mask.cpu().numpy()
        pos_mask = edge_mask_np.copy()
        pos_mask[pos_mask < 0] = 0
        neg_mask = edge_mask_np.copy()
        neg_mask[neg_mask > 0] = 0
        neg_mask = np.abs(neg_mask)
        
        if pos_mask.sum() > 0:
            pos_mask = pos_mask / pos_mask.sum()
        if neg_mask.sum() > 0:
            neg_mask = neg_mask / neg_mask.sum()
        
        normalized_mask = pos_mask - neg_mask
        top_k = min(num_top_edges, len(normalized_mask))
        top_edges_indices = np.argsort(np.abs(normalized_mask))[-top_k:]
        
        for i in reversed(top_edges_indices):
            edge_idx = int(i)
            src_node = edge_index_data[0, edge_idx].item()
            dst_node = edge_index_data[1, edge_idx].item()
            edge_importance = normalized_mask[edge_idx]
            
            edge_attrs = {}
            if edge_attr_data is not None and len(Config.EDGE_FEATURE_NAMES) == Config.EDGE_DIM:
                for attr_idx, attr_name in enumerate(Config.EDGE_FEATURE_NAMES):
                    edge_attrs[attr_name] = edge_attr_data[edge_idx, attr_idx].item()

            top_edges_impact_list.append({
                'source': src_node,
                'destination': dst_node,
                'importance': float(edge_importance),
                'attributes': edge_attrs
            })

    model.eval()
    with torch.no_grad():
        predicted_proba = model(x_data, edge_index_data, edge_attr_data)[node_idx_to_explain].squeeze().item()

    actual_label = None
    if hasattr(graph_data, 'y') and graph_data.y is not None:
        actual_label = graph_data.y[node_idx_to_explain].item()

    gnn_explainer_dict = {
        'node_index': node_idx_to_explain,
        'predicted_proba': predicted_proba,
        'actual_label': actual_label,
        'top_features_impact': top_features_impact,
        'top_edges_impact': top_edges_impact_list
    }

    return gnn_explainer_dict

def _translate_network_to_customer_language(gnn_data):

    top_edges = gnn_data.get('top_edges_impact', [])
    if not top_edges:
        return "you don't have significant connections to other borrowers in our analysis"
    
    current_node = gnn_data.get('node_index', 0)
    area_borrowers = set()
    provider_borrowers = set()
    area_provider_borrowers = set()
    area_risks = []
    provider_risks = []
    area_provider_risks = []
    
    for edge in top_edges:
        attrs = edge['attributes']
        importance = edge['importance']
        src = edge['source']
        dst = edge['destination']
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
        avg_area_risk = sum(area_risks) / len(area_risks)
        risk_descriptor = "concerning" if avg_area_risk < -0.02 else "positive" if avg_area_risk > 0.02 else "neutral"
        borrower_count = len(area_borrowers)
        connection_parts.append(f"you're connected to {borrower_count} other borrower{'s' if borrower_count != 1 else ''} in your geographic area (showing {risk_descriptor} patterns)")
    
    if provider_borrowers:
        avg_provider_risk = sum(provider_risks) / len(provider_risks)
        risk_descriptor = "concerning" if avg_provider_risk < -0.02 else "positive" if avg_provider_risk > 0.02 else "neutral"
        borrower_count = len(provider_borrowers)
        connection_parts.append(f"you share the same lender type with {borrower_count} other borrower{'s' if borrower_count != 1 else ''} (showing {risk_descriptor} patterns)")
    
    if area_provider_borrowers:
        avg_combined_risk = sum(area_provider_risks) / len(area_provider_risks)
        risk_descriptor = "concerning" if avg_combined_risk < -0.02 else "positive" if avg_combined_risk > 0.02 else "neutral"
        borrower_count = len(area_provider_borrowers)
        connection_parts.append(f"you're connected to {borrower_count} other borrower{'s' if borrower_count != 1 else ''} who both live in your area AND use the same lender type (showing {risk_descriptor} patterns)")
    
    if not connection_parts:
        unique_borrowers = set()
        for edge in top_edges:
            src = edge['source']
            dst = edge['destination']
            other_borrower = dst if src == current_node else src
            unique_borrowers.add(other_borrower)
        return f"you have connections to {len(unique_borrowers)} other borrower{'s' if len(unique_borrowers) != 1 else ''} that don't fall into clear geographic or lender categories"
    
    all_unique_borrowers = area_borrowers | provider_borrowers | area_provider_borrowers
    total_borrowers = len(all_unique_borrowers)
    all_importances = [edge['importance'] for edge in top_edges]
    avg_importance = sum(all_importances) / len(all_importances)
    
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
    

def build_deepseek_prompt_for_gnn_explanation(explanation_data, df_row=None):

    predicted_proba = explanation_data['predicted_proba']
    network_context = _translate_network_to_customer_language(explanation_data)
    top_edges = explanation_data.get('top_edges_impact', [])
    has_network_info = bool(top_edges) and not network_context.startswith("you don't have significant connections")
    approval_chance = (1 - predicted_proba) * 100
    if approval_chance > 85:
        approval_context = f"very strong approval chances ({approval_chance:.0f}% likely)"
    elif approval_chance > 65:
        approval_context = f"good approval chances ({approval_chance:.0f}% likely)"
    elif approval_chance > 50:
        approval_context = f"decent approval chances ({approval_chance:.0f}% likely)"
    else:
        approval_context = f"tougher approval process ({approval_chance:.0f}% likely)"
    if has_network_info:
        user_prompt = f""" 
        You are explaining to someone why their loan application has {approval_context} based on their area connections and local lending patterns.
        NETWORK ANALYSIS: {network_context}
        Your {approval_chance:.0f}% approval chance is primarily determined by these area and lender connections, not your individual credit profile. 
        Write a comprehensive explanation that covers:
        → What their {approval_chance:.0f}% approval chance means in practical terms
        → The specific network connections (with exact numbers of other borrowers)
        → Why area connections indicate local job market stability and housing values
        → Why lender connections show lending expertise and risk assessment patterns
        → Specific advice for maintaining or strengthening these network advantages
        → How they compare to other borrowers in similar network situations
        REQUIREMENTS:
        - Focus entirely on network/area factors
        - never mention individual credit factors
        - Always include specific numbers (\"connected to X other borrowers\")
        - Explain the economic reasoning behind network-based lending decisions
        - Provide actionable recommendations for their situation
        - Write a thorough explanation of at least 8 sentences
        - Use terms like "job market," "housing values," "lending expertise," "economic stability"
        Be comprehensive and detailed in your explanation.
        """.strip()
    else:
        top_features = explanation_data.get('top_features_impact', {})
        primary_features = [(f, imp) for f, imp in top_features.items() if f != 'fico']
        if not primary_features and 'fico' in top_features:
            primary_features = [('fico', top_features['fico'])]
        if primary_features:
            main_feature = primary_features[0][0].replace('_', ' ')
            factors_context = f"your {main_feature}"
        else:
            factors_context = "your financial profile"
        user_prompt = f""" 
        You are explaining to someone why their loan application has {approval_context}. 
        Since you don't have significant area connections affecting this decision, it's based on individual factors, but this is unusual for our network-based analysis.
        INDIVIDUAL SITUATION:
        You don't have meaningful connections to other borrowers in your area or through your lender type, which means your application is evaluated more traditionally on individual financial factors rather than network patterns.
        WRITE A DETAILED EXPLANATION following these rules:
        1. Start with what their {approval_chance:.0f}% approval chance means (2 sentences)
        2. Clearly state that area/network connections aren't a major factor (2 sentences)
        3. Explain this makes their case more traditional/individual-focused (2-3 sentences)
        4. Focus on their main individual financial factor without overemphasizing FICO (2-3 sentences)
        5. Give specific improvement advice for their situation (2-3 sentences) - Recommendations for strengthening individual profile, improving key factors
        6. End with overall perspective and next steps (2 sentences)
        REQUIREMENTS:
        - Write 8-12 sentences for comprehensive coverage
        - Acknowledge the lack of network effects prominently
        - Focus on the most important individual factor available
        - Do not mention FICO or credit scores
        - Give concrete advice for their individual situation
        - Compare individual vs network-based evaluations
        TONE: Like a financial advisor explaining why this case is evaluated differently due to lack of network connections.
        """.strip()
    return GLOBAL_SYSTEM_PROMPT, user_prompt

def initialize_deepseek():

    print("Loading fine-tuned DeepSeek-R1 model...")
    
    try:
        config = DeepSeekGATConfig()
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        print(f"Loading base model with 8-bit quantization")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL_PATH
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading LoRA adapter: {config.DEEPSEEK_MODEL_PATH}")
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
            max_length=4000,
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
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
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
            "COMBINED BORROWER-LEVEL",
            "COMBINED INSTRUCTIONAL",
            "CHAIN-OF-THOUGHT",
            "DISAMBIGUATION HINT",
            "\n\nmodel",
            "\nmodel",
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

    
def load_selected_indices_and_map_to_graph(indices_file_path, graph_data):

    try:
        with open(indices_file_path, 'r') as f:
            selected_test_indices = json.load(f)
        
        print(f"Loaded {len(selected_test_indices)} indices from XGB pipeline")
        
        mapped_node_indices = []
        for test_idx in selected_test_indices:
            if test_idx < graph_data.num_nodes:
                mapped_node_indices.append(test_idx)
        
        print(f"Successfully mapped {len(mapped_node_indices)} indices to graph nodes")
        return mapped_node_indices
        
    except FileNotFoundError:
        print(f"Indices file not found at {indices_file_path}")
        print("Falling back to random sampling...")
        return None
    except Exception as e:
        print(f"Error loading indices: {e}")
        return None
    
def load_selected_indices_and_map_to_graph(indices_file_path, graph_data):

    try:
        with open(indices_file_path, 'r') as f:
            selected_test_indices = json.load(f)
        
        print(f"Loaded {len(selected_test_indices)} indices from XGB pipeline")

        mapped_node_indices = []
        for test_idx in selected_test_indices:
            if test_idx < graph_data.num_nodes:
                mapped_node_indices.append(test_idx)
        
        print(f"Successfully mapped {len(mapped_node_indices)} indices to graph nodes")
        return mapped_node_indices

    except Exception as e:
        print(f"Error loading indices: {e}")
        return None

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

if __name__ == '__main__':
    
    random.seed(42)

    try:
        deepseek_model, deepseek_tokenizer, success = initialize_deepseek()
        if not success:
            print("Failed to initialize DeepSeek model. Exiting...")
            sys.exit(1)
            
        trained_model = load_trained_gat_model(
            Config.TRAINED_MODEL_PATH,
            Config.IN_CHANNELS,
            Config.HIDDEN_CHANNELS,
            Config.OUT_CHANNELS,
            Config.HEADS,
            Config.EDGE_DIM
        )

        july_graph_data = load_graph_data(Config.JULY_GRAPH_PATH)
        
        xgb_indices_file = "../evaluations/gemini_xgb/selected_borrower_indices.json"
        selected_node_indices = load_selected_indices_and_map_to_graph(xgb_indices_file, july_graph_data)
        
        if selected_node_indices is None:
            print("Using random sampling for node selection...")
            selected_node_indices = []
            processed_node_indices = set()
            while len(selected_node_indices) < Config.NUM_SAMPLES:
                node_idx = random.randint(0, july_graph_data.num_nodes - 1)
                if node_idx not in processed_node_indices:
                    selected_node_indices.append(node_idx)
                    processed_node_indices.add(node_idx)
        else:
            Config.NUM_SAMPLES = len(selected_node_indices)
            print(f"Updated NUM_SAMPLES to {Config.NUM_SAMPLES} to match XGB pipeline")

        SAVE_PATH = Path(Config.EXPLANATIONS_SAVE_DIR) / Config.EXPLANATIONS_FILE_NAME
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

        all_perplexities = []
        all_individual_feature_coverage_scores = []
        all_individual_feature_consistency_scores = []
        all_network_coverage_scores = []
        all_network_consistency_scores = []
        
        for i, node_idx_to_explain in enumerate(selected_node_indices):
            print(f"\n### Processing Sample {i+1}/{len(selected_node_indices)} for node index: {node_idx_to_explain} ###")

            explanation_results = get_gnnexplainer_insights_for_node(
                trained_model,
                july_graph_data,
                node_idx_to_explain,
                Config.NUM_TOP_EXPLAIN_FEATURES,
                Config.NUM_TOP_EXPLAIN_EDGES
            )
                
            df = pd.read_csv("../data/cleaned_data/df_origination_train_scaled.csv", index_col='id')

            df_row = None
            if df is not None:
                try:
                    if node_idx_to_explain < len(df):
                        df_row = df.iloc[node_idx_to_explain]
                        print(f"Using dataframe row {node_idx_to_explain} for feature translations")
                        df_row = None
                except Exception as e:
                    print(f"Warning: Could not get dataframe row for node {node_idx_to_explain}: {e}")
                    df_row = None

            system_prompt, user_prompt = build_deepseek_prompt_for_gnn_explanation(explanation_results, df_row)
            deepseek_response = get_deepseek_response(system_prompt, user_prompt, deepseek_model, deepseek_tokenizer)

            print("\n### DeepSeek's Explanation ###")
            print(deepseek_response)

            perplexity = None
            try:
                print("\nCalculating perplexity...")
                perplexity = calculate_perplexity_with_gpt_neo(deepseek_response)
                if perplexity is not None:
                    print(f"Perplexity: {perplexity:.2f}")

            except Exception as e:
                print(f"Error during perplexity calculation: {e}")

            grades = None
            try:
                print("\nGrading explanation with LLM-as-a-Judge...")
                grades = llm_grade_explanation(deepseek_response, explanation_results, model_type="gnn")
                if grades:
                    print(f"LLM-as-a-Judge Scores: {grades}")
            except Exception as e:
                print(f"Error during LLM grading: {e}")

            if perplexity is not None and grades:
                all_perplexities.append(perplexity)
                all_individual_feature_coverage_scores.append(grades["Individual Feature Coverage"])
                all_individual_feature_consistency_scores.append(grades["Individual Feature Consistency"])
                all_network_coverage_scores.append(grades["Network Coverage"])
                all_network_consistency_scores.append(grades["Network Consistency"])

                entry = {
                    "node_index": node_idx_to_explain,
                    "explanation": deepseek_response
                }

                with open(SAVE_PATH, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                print(f"Explanation for node {node_idx_to_explain} saved to {SAVE_PATH}")
            else:
                print(f"Not saving explanation for node {node_idx_to_explain} due to missing perplexity or grades.")

        print("\n### Evaluation Summary ###\n")

        print(f"\n### Sample Sizes ###")
        print(f"Perplexity: {len(all_perplexities)} samples")
        print(f"Individual Feature Coverage: {len(all_individual_feature_coverage_scores)} samples")
        print(f"Individual Feature Consistency: {len(all_individual_feature_consistency_scores)} samples")
        print(f"Network Coverage: {len(all_network_coverage_scores)} samples")
        print(f"Network Consistency: {len(all_network_consistency_scores)} samples")

        min_samples = min(len(all_perplexities), len(all_individual_feature_coverage_scores), 
                          len(all_individual_feature_consistency_scores), len(all_network_coverage_scores),
                          len(all_network_consistency_scores))

        print("\n### 95% Confidence Intervals ###")
        for name, data in [
            ("Perplexity", all_perplexities),
            ("Individual Feature Coverage", all_individual_feature_coverage_scores),
            ("Individual Feature Consistency", all_individual_feature_consistency_scores),
            ("Network Coverage", all_network_coverage_scores),
            ("Network Consistency", all_network_consistency_scores)
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

            elif name.startswith(("Individual", "Network")) and len(data) > 1 and np.std(data) > 0:
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

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()