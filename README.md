# AmazonHelp AI Support Agent

An AI-powered customer-support agent built for the **Hiver SDE Intern Take-Home Assignment** using the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.

> **Goal:** Turn historical customer-support conversations into a measurable, trustworthy support agent — with evaluation as a first-class component.

---

## What It Does

Given an incoming customer message, the agent:

1. **Classifies intent** using a brand-specific taxonomy derived from the data.
2. **Retrieves similar historical interactions** to ground its response.
3. **Drafts a concise support reply** consistent with historical brand behavior.
4. **Decides whether to auto-handle or escalate** the conversation.
5. **Explains every escalation decision.**

```text
Customer Message
       │
       ▼
Intent Classification
       │
       ▼
Historical Retrieval
       │
       ▼
Reply Generation
       │
       ▼
Escalation Decision
       │
       ▼
Reply + Decision + Reason

Selected Brand

AmazonHelp

AmazonHelp was selected after dataset-level analysis based on the volume and availability of customer-support interactions, providing sufficient data for both historical retrieval and rigorous evaluation.

Evaluation

The system is evaluated against:

Majority-class baseline — trivial baseline
TF-IDF + Logistic Regression — classical ML baseline
AI support agent — proposed system

Evaluation covers:

Intent accuracy and Macro-F1
Per-intent performance
Escalation precision, recall and F1
Safe auto-handling performance
Reply correctness and groundedness
Hallucination / unsupported-claim rate
LLM-as-judge quality
Human vs. LLM-judge agreement
Failure analysis

A 150–250 example human-labelled golden set is used as the primary evaluation set.

Dataset

Customer Support on Twitter

Kaggle identifier:

thoughtvector/customer-support-on-twitter

The raw dataset is not committed to this repository because of its size.

Expected local location:

data/raw/twcs/twcs.csv

See data/README.md for setup instructions.

Repository Structure
hiver-sde-takehome/
│
├── data/            # Dataset documentation and local data
├── notebooks/       # Exploration and analysis
├── src/             # Agent implementation
│   ├── data/
│   ├── intents/
│   ├── retrieval/
│   ├── generation/
│   └── escalation/
│
├── baselines/       # Baseline implementations
├── evaluation/      # Metrics, judge and evaluation harness
├── golden/          # Human-labelled evaluation set
├── report/          # Final report and analysis
└── tests/           # Automated tests
Quick Start
1. Clone
git clone <https://github.com/pithiyakeval/hiver-sde-takehome>
cd hiver-sde-takehome
2. Create environment
python -m venv .venv

Windows:

.\.venv\Scripts\Activate.ps1

macOS/Linux:

source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment

Copy .env.example to .env and add the required LLM configuration.

5. Run evaluation
python -m evaluation.evaluate

The final repository will reproduce the headline evaluation results using a small committed evaluation fixture without requiring the full raw dataset.

Design Principles
Evidence over demos

Performance is measured on a held-out human-labelled set rather than a handful of manually selected examples.

Grounded generation

Replies are generated using historically similar support interactions rather than relying solely on the model's general knowledge.

Conservative automation

Cases requiring account-specific actions, sensitive investigation, or insufficient evidence should be escalated rather than confidently answered.

Honest reporting

Headline metrics are accompanied by failure analysis and a discussion of what those numbers may hide.

Project Status

Current phase: Dataset exploration → intent discovery → evaluation design

The implementation and reported metrics will be updated as the project progresses.

License & Data

This repository contains project code and evaluation artifacts.

The original Twitter support dataset is not redistributed here. Please obtain it directly from the dataset source and comply with its applicable terms of use.


## One important point

This is the **right README for the current stage**.

We should **not put fake results in it yet**. Later, once we actually run the experiments, we'll replace the evaluation section with something like:

```text
| System | Intent Macro-F1 | Escalation F1 | Reply Quality |
|---|---:|---:|---:|
| Majority | actual | — | — |
| TF-IDF + LR | actual | — | — |
| AI Agent | actual | actual | actual |
