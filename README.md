# Credit Risk Explainability with LLMs

## Overview
This repository supports the implementation of the paper titled _"Explainability in Credit Risk Modelling: A Comparative Study of Network-based
and Non-network-based Approaches"_. The project explores how structured model explanations—derived from SHAP (for tabular/XGBoost models) and GNNExplainer (for GNNs)—can be translated into human-readable narratives using Large Language Models (LLMs) such as Gemma, DeepSeek, and Gemini.

The primary goal is to evaluate and compare these LLM-generated explanations in terms of clarity, interpretability, and domain relevance across user groups.

---

## Project Structure

```bash
├── LICENSE                     # official license
├── README.md                   # readme
├── chatgpt_simulated_ratings/  # ChatGPT-4o simulated evaluation framework for CRP and NCRP personas                       
├── data_preprocessing/         # Data prep and cleaning
├── finetuned_llms/             # Quantized LoRA fine-tuning for Gemma 3 4B and DeepSeek R1 70B
├── graph_constructions/        # Network construction
└── models/                     # XGBoost, GAT, and bimodal prediction pipelines with explanation generation
```


---
