"""
===============================================================================
Fine-Tuning Dataset Generator for Loan Default Explanations With ChatGPT-4o
===============================================================================

This notebook orchestrates the generation of high-quality, instruction-following training samples for LLM fine-tuning. It integrates predictions and explanations from two distinct models—an XGBoost model using SHAP for tabular feature attribution and a GNN (GAT-based) model using GNNExplainer for relational insights—into ChatGPT-formatted multi-turn conversations. Each sample pairs a structured prompt with a tailored explanation describing why a loan is predicted to default or not. The resulting JSONL outputs support support SHAP-only, GNN-only, or hybrid explanation configurations. The pipeline supports batching, fault-tolerant retries, confidence-based weighting, and prompt reasoning techniques.

Note: This script is intended for academic reference only.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import joblib

import shap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer as GNNExplainerAlgorithm

import openai
from openai import OpenAI, OpenAIError, RateLimitError, APIError
import backoff

from dotenv import load_dotenv

import os
import json
import time
import random
import re
import sys

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

class GNNConfig:
    
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


def calculate_shap_values(df, xgb_model):

    if hasattr(xgb_model, 'feature_names_in_'):
        feature_columns = list(xgb_model.feature_names_in_)
    else:
        exclude_cols = ['default', 'year', 'month', 'id_loan', 'id']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_columns].copy()
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    X = X.fillna(0)
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = shap_values
    
    shap_df = pd.DataFrame(shap_array, columns=feature_columns, index=df.index)
    
    return shap_df

def create_graph_data_from_df(df):
    
    required_cols = GNNConfig.NODE_FEATURE_NAMES + ['provider', 'area', 'default']    
    x = torch.tensor(df[GNNConfig.NODE_FEATURE_NAMES].values, dtype=torch.float32)
    edge_index = []
    edge_attr = []
    node_map = {loan_id: i for i, loan_id in enumerate(df.index)}
    edge_set = set()
    provider_sizes = df.groupby('provider').size()
    area_sizes = df.groupby('area').size()
    max_group_size = max(provider_sizes.max(), area_sizes.max(), 1)
    provider_groups = df.groupby('provider')
    
    print(f"Total edges created: {len(edge_index)//2 if len(edge_index) > 0 else 0}")
    print(f"Unique providers: {df['provider'].nunique()}")
    print(f"Unique areas: {df['area'].nunique()}")
    
    for provider, group in provider_groups:
        if len(group) > 1:
            loan_ids = group.index.tolist()
            group_size = len(group)
            normalized_group_weight = group_size / max_group_size
            
            for i in range(len(loan_ids)):
                for j in range(i+1, len(loan_ids)):
                    src = node_map[loan_ids[i]]
                    dst = node_map[loan_ids[j]]
                    same_area = group.iloc[i]['area'] == group.iloc[j]['area']
                    
                    if same_area:
                        attr = [0.0, 0.0, 1.0, normalized_group_weight]
                    else:
                        attr = [0.0, 1.0, 0.0, normalized_group_weight]
    
                    if (src, dst) not in edge_set and (dst, src) not in edge_set:
                        edge_index.extend([[src, dst], [dst, src]])
                        edge_attr.extend([attr, attr])
                        edge_set.add((src, dst))
    
    area_groups = df.groupby('area')
    
    for area, group in area_groups:
        if len(group) > 1:
            loan_ids = group.index.tolist()
            group_size = len(group)
            normalized_group_weight = group_size / max_group_size
            
            for i in range(len(loan_ids)):
                for j in range(i+1, len(loan_ids)):
                    src = node_map[loan_ids[i]]
                    dst = node_map[loan_ids[j]]
                    different_provider = group.iloc[i]['provider'] != group.iloc[j]['provider']
                    
                    if different_provider and (src, dst) not in edge_set and (dst, src) not in edge_set:
                        attr = [1.0, 0.0, 0.0, normalized_group_weight]
                        edge_index.extend([[src, dst], [dst, src]])
                        edge_attr.extend([attr, attr])
                        edge_set.add((src, dst))
    
    if len(edge_index) == 0:
        print("Warning: Created graph with no edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, GNNConfig.EDGE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    if edge_attr.shape[1] != GNNConfig.EDGE_DIM:
        raise ValueError(f"Edge attributes have {edge_attr.shape[1]} dimensions, expected {GNNConfig.EDGE_DIM}")
        
    y = torch.tensor(df['default'].values, dtype=torch.float32).view(-1, 1)
    
    if edge_index.shape[1] > 0:
        edge_attr_np = edge_attr.numpy()
        area_edges = np.sum(edge_attr_np[:, 0])  
        provider_edges = np.sum(edge_attr_np[:, 1])   
        area_provider_edges = np.sum(edge_attr_np[:, 2])
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def get_gnnexplainer_insights_for_node(model, graph_data, node_idx_to_explain, num_top_features, num_top_edges):

    model_config = {
        "mode": "binary_classification",
        "task_level": "node",
        "return_type": "probs"
    }

    algorithm = GNNExplainerAlgorithm(
        epochs=GNNConfig.EXPLAINER_EPOCHS,
        lr=GNNConfig.EXPLAINER_LR
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
    feature_names = GNNConfig.NODE_FEATURE_NAMES

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
            if edge_attr_data is not None and len(GNNConfig.EDGE_FEATURE_NAMES) == GNNConfig.EDGE_DIM:
                for attr_idx, attr_name in enumerate(GNNConfig.EDGE_FEATURE_NAMES):
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

def generate_gnn_explanations(df, gnn_model, graph_data, num_samples=None, specific_loan_id=None):

    explanations = []
    
    if specific_loan_id is not None:
        node_indices = [df.index.get_loc(specific_loan_id)]
    else:
        node_indices = range(min(len(df), num_samples) if num_samples else len(df))

    successful_explanations = 0
    for idx in node_indices:
        try:
            explanation = get_gnnexplainer_insights_for_node(
                gnn_model, graph_data, idx,
                GNNConfig.NUM_TOP_EXPLAIN_FEATURES,
                GNNConfig.NUM_TOP_EXPLAIN_EDGES
            )
            
            if explanation is not None:
                loan_id = df.index[idx]
                explanations.append({
                    "id": str(loan_id),
                    "predicted_proba": explanation['predicted_proba'],
                    "top_features_impact": explanation['top_features_impact'],
                    "top_edges_impact": explanation['top_edges_impact']
                })
                successful_explanations += 1
            
        except Exception as e:
            print(f"Failed to explain node {idx}: {e}")
            continue
            
    return explanations

def generate_shap_explanation_prompt(df, df_row, shap_row, xgb_model):
    
    loan_id = df_row.name
    feature_columns = list(xgb_model.feature_names_in_) if hasattr(xgb_model, 'feature_names_in_') else [col for col in df.columns if col not in ['default', 'year', 'month', 'id_loan', 'id']]
    X_row = df_row[feature_columns].values.reshape(1, -1)
    prediction_proba = xgb_model.predict_proba(X_row)[0][1]
    credit_score_explanation = _get_credit_score_explanation(df_row, df_row.get('fico', 0.5))
    top_shap_features = shap_row.abs().nlargest(5)  
    customer_factors = []
    improvement_advice = []
    top_factors_formatted = [f"1. {credit_score_explanation}"]
    factor_count = 2 
    
    for feature, impact_abs in top_shap_features.items():
        if feature == 'fico':
            continue
            
        feature_value = df_row.get(feature, 'N/A')
        plain_name, pop_context, impact_level, direction, advice = _translate_features_to_customer_language(
            feature, feature_value, shap_row[feature]
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

    approval_chance = (1 - prediction_proba) * 100
    if approval_chance > 85:
        approval_context = f"excellent approval chances ({approval_chance:.0f}% likely to be approved)"
    elif approval_chance > 65:
        approval_context = f"good approval chances ({approval_chance:.0f}% likely to be approved)"
    elif approval_chance > 50:
        approval_context = f"fair approval chances ({approval_chance:.0f}% likely to be approved)"
    else:
        approval_context = f"challenging approval odds ({approval_chance:.0f}% likely to be approved)"
    
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
    
    return {
        "system": GLOBAL_SYSTEM_PROMPT,
        "user": user_prompt
    }

def generate_gnn_explanation_prompt(df, df_row, gnn_data):
    
    current_loan_id = df_row.name
    predicted_proba = gnn_data["predicted_proba"]
    network_context = _translate_network_to_customer_language(gnn_data)
    top_features = gnn_data.get('top_features_impact', {})
    customer_factors = []
    salient_features_formatted = []
    
    for i, (feature, importance) in enumerate(list(top_features.items())[:3]):
        feature_value = df_row.get(feature, 0)
        if feature in ['fico', 'dti', 'ltv', 'orig_upb', 'loan_term']:
            plain_name, pop_context, impact_level, direction, advice = _translate_features_to_customer_language(
                feature, feature_value, importance
            )
            customer_factors.append(f"your {plain_name} {pop_context}")
            salient_features_formatted.append(f"{i+1}. Your {plain_name} {pop_context}")
    
    factors_context = " and ".join(customer_factors[:2]) if customer_factors else "your financial situation looks solid"
    salient_features_list = "\n".join(salient_features_formatted[:2])
    
    approval_chance = (1 - predicted_proba) * 100
    if approval_chance > 85:
        approval_context = f"very strong approval chances ({approval_chance:.0f}% likely)"
    elif approval_chance > 65:
        approval_context = f"good approval chances ({approval_chance:.0f}% likely)"
    elif approval_chance > 50:
        approval_context = f"decent approval chances ({approval_chance:.0f}% likely)"
    else:
        approval_context = f"tougher approval process ({approval_chance:.0f}% likely)"
    
    user_prompt = f"""
You are explaining to someone why their loan application has {approval_context}, looking at both their personal finances and their local area.

PERSONAL SITUATION:
{factors_context}.

LOCAL CONTEXT:
When we look at your area and similar situations, {network_context}.

KEY FACTORS:
{salient_features_list}
Area patterns: {network_context}

WRITE AN EXPLANATION that a smart 17-year-old would understand. Follow these rules:

1. Start with what their {approval_chance:.0f}% approval chance means
2. Explain their strongest personal factor and why banks care about it
3. Explain how their location/area affects things in simple terms
4. Show how personal + area factors work together
5. Give specific advice for improvement
6. End with perspective on their overall situation

REQUIREMENTS:
- Never use terms like "GNN," "network analysis," or "loan-to-value"
- Always include SPECIFIC NUMBERS when mentioning connections (e.g., "(number) other borrowers," not "a couple of folks")
- Compare to "other people your age," "people in your area," "typical borrowers"
- Connection counts are important data points that help explain the decision - include them exactly
- Explain WHY location matters (economic conditions, local job market, etc.)
- Sound like a helpful financial advisor explaining to a teenager
- Keep to 6-8 sentences total

TONE: Encouraging and educational, like a teacher explaining how local economics affects personal finance.
"""
    
    return {
        "system": GLOBAL_SYSTEM_PROMPT,
        "user": user_prompt
    }

def get_hybrid_prediction(pred_xgb, pred_gnn, max_conf=0.4, disagreement_thresh=0.3):

    raw_xgb_conf = abs(pred_xgb - 0.5)
    raw_gnn_conf = abs(pred_gnn - 0.5)
    xgb_conf = min(raw_xgb_conf, max_conf)
    gnn_conf = min(raw_gnn_conf, max_conf)
    disagreement = abs(pred_xgb - pred_gnn)
    
    if disagreement > disagreement_thresh:
        xgb_conf *= 0.5
        gnn_conf *= 0.5
    
    total_conf = xgb_conf + gnn_conf
    if total_conf > 0:
        hybrid_pred = (pred_xgb * xgb_conf + pred_gnn * gnn_conf) / total_conf
    else:
        hybrid_pred = (pred_xgb + pred_gnn) / 2 
    
    return {
        'hybrid_prediction': hybrid_pred,
        'xgb_weight': xgb_conf / total_conf if total_conf > 0 else 0.5,
        'gnn_weight': gnn_conf / total_conf if total_conf > 0 else 0.5,
        'disagreement': disagreement,
        'raw_xgb_confidence': raw_xgb_conf,
        'raw_gnn_confidence': raw_gnn_conf,
        'capped_xgb_confidence': xgb_conf,
        'capped_gnn_confidence': gnn_conf,
        'agreement_status': 'agree' if (pred_xgb - 0.5) * (pred_gnn - 0.5) > 0 else 'disagree'
    }

def generate_hybrid_explanation_prompt(df, df_row, shap_row, gnn_data, xgb_model):
    
    feature_columns_xgb = list(xgb_model.feature_names_in_) if hasattr(xgb_model, 'feature_names_in_') else [col for col in df.columns if col not in ['default', 'year', 'month', 'id_loan', 'id']]
    X_row_xgb = df_row[feature_columns_xgb].values.reshape(1, -1)
    prediction_proba_xgb = xgb_model.predict_proba(X_row_xgb)[0][1]
    predicted_proba_gnn = gnn_data.get("predicted_proba", 0.5)
    hybrid_results = get_hybrid_prediction(prediction_proba_xgb, predicted_proba_gnn)
    overall_predicted_proba = hybrid_results['hybrid_prediction']
    credit_score_explanation = _get_credit_score_explanation(df_row, df_row.get('fico', 0.5))
    top_shap_features = shap_row.abs().nlargest(4)
    personal_factors = []
    improvement_advice = []
    individual_factors_formatted = [f"1. {credit_score_explanation}"]
    
    factor_count = 2
    for feature, impact_abs in top_shap_features.items():
        if feature == 'fico':
            continue
            
        feature_value = df_row.get(feature, 'N/A')
        if feature in ['dti', 'ltv', 'orig_upb', 'loan_term']:
            plain_name, pop_context, impact_level, direction, advice = _translate_features_to_customer_language(
                feature, feature_value, shap_row[feature]
            )
            personal_factors.append(f"your {plain_name} {pop_context}")
            individual_factors_formatted.append(f"{factor_count}. Your {plain_name} {pop_context}")
            
            if "hurts" in direction and impact_level in ["really", "significantly"]:
                improvement_advice.append(advice)
                
            factor_count += 1
            if factor_count > 3:
                break
    
    factors_text = " and ".join(personal_factors[:2])
    advice_text = improvement_advice[0] if improvement_advice else "keep managing your finances well"
    individual_factors_list = "\n".join(individual_factors_formatted[:3])
    network_context = _translate_network_to_customer_language(gnn_data)
    
    approval_chance = (1 - overall_predicted_proba) * 100
    if approval_chance > 85:
        approval_context = f"excellent approval chances ({approval_chance:.0f}% likely to get approved)"
    elif approval_chance > 65:
        approval_context = f"strong approval chances ({approval_chance:.0f}% likely to get approved)"
    elif approval_chance > 50:
        approval_context = f"reasonable approval chances ({approval_chance:.0f}% likely to get approved)"
    else:
        approval_context = f"challenging approval process ({approval_chance:.0f}% likely to get approved)"
    
    user_prompt = f"""
You are explaining to someone why their loan application has {approval_context} after looking at everything - their personal finances, their local area, and similar borrowers.

{credit_score_explanation}.

COMPLETE PICTURE:
Personal finances: {factors_text}
Local context: {network_context}

BREAKDOWN:
Personal factors:
{individual_factors_list}

Area factors: {network_context}

WRITE AN EXPLANATION that a smart 17-year-old would understand. Follow these rules:

1. Start with what their {approval_chance:.0f}% approval chance means overall
2. Always mention their credit score situation first
3. Explain their most important personal financial factor after credit score
4. Explain how their area/location adds context
5. Show how personal + area factors combine to create their final result
6. Give specific advice: {advice_text}
7. Mention how they could improve both personal and situational factors
8. End with encouraging perspective about their complete picture

REQUIREMENTS:
- Never use banking jargon, model names, or technical terms
- Always include SPECIFIC NUMBERS when mentioning connections (e.g., "(number) other borrowers," not "a couple of folks")
- Compare to "other people your age," "people in your area," "typical borrowers"
- Connection counts are important data points that help explain the decision - include them exactly
- Explain WHY each factor matters in real life
- Give concrete timelines and improvement amounts
- Sound like a helpful mentor teaching about money and location
- Keep to 7-9 sentences total

TONE: Encouraging teacher explaining how personal finance and local economics work together.
"""
    
    return {
        "system": GLOBAL_SYSTEM_PROMPT,
        "user": user_prompt
    }

def load_gnn_model_and_data(gnn_model_path, df):

    try:
        model = GAT(
            in_channels=GNNConfig.IN_CHANNELS,
            hidden_channels=GNNConfig.HIDDEN_CHANNELS,
            out_channels=GNNConfig.OUT_CHANNELS,
            heads=GNNConfig.HEADS,
            edge_dim=GNNConfig.EDGE_DIM
        )
        
        print(f"Loading GAT model from: {gnn_model_path}")
        model.load_state_dict(torch.load(gnn_model_path))
        model.eval()
        print("GAT model loaded successfully!")
        
        graph_data = create_graph_data_from_df(df)
        return model, graph_data
    except Exception as e:
        print(f"Failed to load GNN model or create graph data: {e}")
        return None, None

def flag_counterintuitive_explanations(explanation_text):

    patterns = [
        r"excellent.*hurt",
        r"very good.*hurt", 
        r"high credit score.*hurt",
        r"good.*increase.*risk"
    ]
    for pattern in patterns:
        if re.search(pattern, explanation_text, re.IGNORECASE):
            return True
    return False

def clarify_explanation(client, original_prompt, original_explanation, model="gpt-3.5-turbo"):
    
    clarification_prompt = (
        "The explanation seemed confusing - it suggested something good was actually bad. "
        "Please rewrite this to make it clear and logical for someone who isn't familiar with banking. "
        "If there's an unusual situation, explain it in simple terms.\n\n"
        f"Original explanation: {original_explanation}\n\n"
        "Clearer explanation:"
    )
    
    messages = [
        {"role": "system", "content": original_prompt["system"]},
        {"role": "user", "content": clarification_prompt}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        max_tokens=512
    )
    return response.choices[0].message.content

def log_flagged_explanation(log_path, original, clarified):
    
    with open(log_path, 'a') as f:
        f.write(json.dumps({
            'original': original,
            'clarified': clarified
        }) + '\n')

def generate_finetune_dataset_with_chatgpt(api_key: str, data_path: str, xgb_model_path: str, gnn_model_path: str, num_samples: int, output_dir: str, chatgpt_model: str = "gpt-4o", batch_size: int = 20, save_every: int = 10) -> None:
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = load_partial_results(output_dir)
    
    try:
        print("Loading data...")
        df = load_data(data_path, num_samples, results.get('processed_ids', set()))
        
        print("Loading models...")
        xgb_model = load_xgb_model(xgb_model_path)
        gnn_model, graph_data = load_gnn_model_and_data(gnn_model_path, df)
        
        client = OpenAI(api_key=api_key)
        
        print("\nCalculating SHAP values...")
        shap_df = calculate_shap_values(df, xgb_model)
        
        gnn_explanations = {}
        if gnn_model and graph_data:
            print("\nGenerating GNN explanations...")
            gnn_explanations = generate_gnn_explanations_wrapper(df, gnn_model, graph_data)
        
        print("\nGenerating ChatGPT explanations...")
        processed_samples = 0
        
        for i, (loan_id, df_row) in enumerate(df.iterrows()):
            if str(loan_id) in results.get('processed_ids', set()):
                continue
                
            print(f"\nProcessing loan {loan_id} ({i+1}/{len(df)})...")
            
            sample_results = process_single_sample(
                df_row, df, shap_df, gnn_explanations, 
                xgb_model, client, chatgpt_model
            )
            
            for explanation_type in sample_results:
                results[explanation_type].extend(sample_results[explanation_type])
            results.setdefault('processed_ids', set()).add(str(loan_id))
            
            if (i + 1) % save_every == 0:
                save_results_incrementally(results, output_dir)
                print(f"Saved intermediate results after {i+1} samples")
                
            time.sleep(1)
            
        save_results_incrementally(results, output_dir)
        
    except Exception as e:
        print(f"\nCritical error encountered: {e}")
        print("Attempting to save partial results...")
        save_results_incrementally(results, output_dir)
        raise
        
    print_summary(results, len(df))

def load_partial_results(output_dir: str) -> Dict:

    results = {
        'shap': [],
        'gnn': [],
        'combined': [],
        'processed_ids': set()
    }
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        loan_id = data['messages'][1]['content'].split('loan ')[1].split(' ')[0]
                        results['processed_ids'].add(loan_id)
            except Exception as e:
                print(f"Warning: Could not load partial results from {filename}: {e}")
                
    return results

def load_data(data_path: str, num_samples: int, processed_ids: set) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path, index_col='id')
        unprocessed_df = df[~df.index.astype(str).isin(processed_ids)]
        if len(unprocessed_df) == 0:
            print("No new unprocessed loans remaining.")
            return pd.DataFrame()
        return unprocessed_df.sample(n=min(num_samples, len(unprocessed_df)), random_state=random.randint(0, 99999))
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def load_xgb_model(xgb_model_path: str):

    try:
        print(f"Loading XGB model from: {xgb_model_path}")
        model = joblib.load(xgb_model_path)
        print(f"XGBoost model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load XGBoost model: {e}")

def generate_gnn_explanations_wrapper(df: pd.DataFrame, gnn_model, graph_data) -> Dict:

    try:
        explanations = generate_gnn_explanations(df, gnn_model, graph_data)
        return {item['id']: item for item in explanations}
    except Exception as e:
        print(f"Warning: Failed to generate GNN explanations: {e}")
        return {}

def process_single_sample(df_row: pd.Series, df: pd.DataFrame, shap_df: pd.DataFrame, gnn_explanations: Dict, xgb_model, client: OpenAI, chatgpt_model: str) -> Dict:
    
    loan_id = df_row.name
    str_loan_id = str(loan_id)
    results = {'shap': [], 'gnn': [], 'combined': []}
    has_shap = loan_id in shap_df.index
    has_gnn = str_loan_id in gnn_explanations
    
    if has_shap:
        try:
            shap_result = generate_shap_explanation(
                df_row, df, shap_df.loc[loan_id], 
                xgb_model, client, chatgpt_model
            )
            if shap_result:
                results['shap'].append(shap_result)
        except Exception as e:
            print(f"Error generating SHAP explanation for loan {loan_id}: {e}")
    
    if has_gnn:
        try:
            gnn_result = generate_gnn_explanation(
                df_row, df, gnn_explanations[str_loan_id],
                client, chatgpt_model
            )
            if gnn_result:
                results['gnn'].append(gnn_result)
        except Exception as e:
            print(f"Error generating GNN explanation for loan {loan_id}: {e}")
    
    if has_shap and has_gnn:
        try:
            combined_result = generate_combined_explanation(
                df_row, df, shap_df.loc[loan_id],
                gnn_explanations[str_loan_id], xgb_model,
                client, chatgpt_model
            )
            if combined_result:
                results['combined'].append(combined_result)
        except Exception as e:
            print(f"Error generating combined explanation for loan {loan_id}: {e}")
    
    return results

def generate_shap_explanation(df_row: pd.Series, df: pd.DataFrame, shap_row: pd.Series, xgb_model, client: OpenAI,  chatgpt_model: str) -> Optional[Dict]:

    shap_prompt = generate_shap_explanation_prompt(df, df_row, shap_row, xgb_model)
    shap_explanation = generate_chatgpt_explanation(client, shap_prompt, model=chatgpt_model)
    
    if flag_counterintuitive_explanations(shap_explanation):
        shap_explanation = clarify_explanation(client, shap_prompt, shap_explanation)
    
    return {
        'messages': [
            {"role": "system", "content": shap_prompt["system"]},
            {"role": "user", "content": shap_prompt["user"]},
            {"role": "assistant", "content": shap_explanation}
        ]
    }

def generate_gnn_explanation(df_row: pd.Series, df: pd.DataFrame, gnn_data: Dict, client: OpenAI, chatgpt_model: str) -> Optional[Dict]:

    gnn_prompt = generate_gnn_explanation_prompt(df, df_row, gnn_data)
    gnn_explanation = generate_chatgpt_explanation(client, gnn_prompt, model=chatgpt_model)
    
    if flag_counterintuitive_explanations(gnn_explanation):
        gnn_explanation = clarify_explanation(client, gnn_prompt, gnn_explanation)
    
    return {
        'messages': [
            {"role": "system", "content": gnn_prompt["system"]},
            {"role": "user", "content": gnn_prompt["user"]},
            {"role": "assistant", "content": gnn_explanation}
        ]
    }

def generate_combined_explanation(df_row: pd.Series, df: pd.DataFrame, shap_row: pd.Series, gnn_data: Dict, xgb_model, client: OpenAI, chatgpt_model: str) -> Optional[Dict]:
    
    combined_prompt = generate_hybrid_explanation_prompt(
        df, df_row, shap_row, gnn_data, xgb_model
    )
    combined_explanation = generate_chatgpt_explanation(
        client, combined_prompt, model=chatgpt_model
    )
    
    if flag_counterintuitive_explanations(combined_explanation):
        combined_explanation = clarify_explanation(
            client, combined_prompt, combined_explanation
        )
    
    return {
        'messages': [
            {"role": "system", "content": combined_prompt["system"]},
            {"role": "user", "content": combined_prompt["user"]},
            {"role": "assistant", "content": combined_explanation}
        ]
    }

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


def save_results_incrementally(results: Dict, output_dir: str) -> None:
    
    for explanation_type in ['shap', 'gnn', 'combined']:
        if results.get(explanation_type):
            filename = f"{explanation_type}_test.jsonl"
            filepath = os.path.join(output_dir, filename)
            
            existing_lines = set()
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing_lines = set(f.read().splitlines())
            
            with open(filepath, 'a') as f:
                for item in results[explanation_type]:
                    line = json.dumps(item)
                    if line not in existing_lines:
                        f.write(line + '\n')

def print_summary(results: Dict, total_samples: int) -> None:

    print("\n### Summary ###")
    print(f"Total loans processed: {total_samples}")
    print(f"SHAP explanations generated: {len(results['shap'])}")
    print(f"GNN explanations generated: {len(results['gnn'])}")
    print(f"Combined explanations generated: {len(results['combined'])}")

@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5)
def generate_chatgpt_explanation(client: OpenAI, prompt_data: Dict, model: str = "gpt-3.5-turbo", max_retries: int = 3, delay: float = 1.0) -> str:

    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": prompt_data["user"]}
                ],
                temperature=0.6,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            if retry == max_retries - 1:
                return f"Failed to generate explanation: {str(e)}"
            time.sleep(delay * (retry + 1))


if __name__ == "__main__":
    try:
        generate_finetune_dataset_with_chatgpt(
            api_key=os.getenv("OPENAI_API_KEY"),
            data_path='../data/cleaned_data/df_origination_train_scaled.csv',
            xgb_model_path='../models/trained_xgb_model.pkl',
            gnn_model_path='../models/trained_gat_model.pt',
            num_samples=1000,
            output_dir="samples",
            chatgpt_model="gpt-4o",
            save_every=20
        )
    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)