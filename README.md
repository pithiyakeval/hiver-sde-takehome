# Hiver SDE Take-Home — AI Support Agent

An evaluation-driven AI customer-support agent built on the **Customer Support on Twitter** dataset.

The system classifies incoming customer messages, retrieves historically similar support interactions, drafts a concise customer-facing response grounded in historical resolutions, and decides whether the case should be handled automatically or escalated to a human.

The implementation follows an **evaluation-first** approach: each major component has a measurable baseline, reproducible evaluation artifacts, and documented failure modes.

---

## 1. Problem Overview

Customer-support teams receive large volumes of messages that differ in wording but often represent recurring support problems.

This project explores a lightweight AI support-agent pipeline that can:

1. Identify the customer's intent.
2. Retrieve historically relevant support interactions.
3. Generate a concise response grounded in historical resolutions.
4. Decide whether the request is suitable for automated handling or should be escalated to a human.

### Selected Brand

**AmazonHelp**

The Customer Support on Twitter dataset contains support interactions from multiple brands. `AmazonHelp` was selected as the single support account used for this project.

The resulting conversation corpus contains approximately **168K direct customer → AmazonHelp response pairs**.

### High-Level Architecture

```text
Customer Message
       │
       ▼
┌──────────────────────┐
│ Intent Classification│
│     Ministral 3B     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Historical Retrieval │
│       TF-IDF         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Response Generation  │
│     Ministral 3B     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Escalation Policy    │
└──────────┬───────────┘
           │
           ▼
      Agent Result

2. Headline Evaluation Results
Evaluation was performed on a 40-example human-labelled frozen evaluation split.
Component	Metric	Result
Intent classification	Accuracy	75.0%
Intent classification	Macro F1	74.38%
Escalation policy	Accuracy	65.0%
LLM-as-judge	Overall	2.77 / 5
LLM-as-judge	Correctness	2.58 / 5
LLM-as-judge	Helpfulness	2.77 / 5
LLM-as-judge	Grounding	4.95 / 5
LLM-as-judge	Safety	5.00 / 5


Important evaluation caveat: The frozen evaluation set was consulted during iterative development. These results should therefore not be interpreted as results from a completely untouched final holdout.

The LLM-as-judge scores are evaluation signals rather than independent human ground truth.
3. Prerequisites
Tool	Version
Git	Any recent version
Python	3.x
Ollama	Latest
Ministral	ministral-3:3b


The generation and LLM-classification components use a local Ollama model, so no external LLM API key is required.
4. Installation
Clone the repository
git clone https://github.com/pithiyakeval/hiver-sde-takehome.git
cd hiver-sde-takehome

Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

macOS / Linux
python -m venv .venv
source .venv/bin/activate

Install dependencies
pip install -r requirements.txt

5. Dataset
This project uses the Kaggle Customer Support on Twitter dataset.
Dataset:
thoughtvector/customer-support-on-twitter

The raw dataset is expected as:
twcs.csv

The dataset is used to:
- identify support accounts
- select AmazonHelp
- reconstruct direct customer → support response pairs
- explore candidate intent structure
- build the historical retrieval corpus
- construct the human-labelled evaluation set
Dataset scale
The original dataset contains approximately:
2.8M tweets
108 support accounts

For this project, the selected AmazonHelp subset contains approximately:
169K AmazonHelp support tweets
168K direct customer → AmazonHelp response pairs
71K unique customer authors

6. Local LLM Setup
The LLM-powered agent uses:
ministral-3:3b

Verify Ollama
ollama list

Pull the model
If it is not already installed:
ollama pull ministral-3:3b

Verify the model
ollama run ministral-3:3b "Say OK"

The application connects to the local Ollama server at:
http://127.0.0.1:11434

No API key is required.
7. Intent Taxonomy
The taxonomy was derived from the selected AmazonHelp customer messages using unsupervised topic exploration followed by manual review.
The final evaluation taxonomy contains 13 intents.
Code	Intent	Description
OS	order_status	Customer asks where an order currently is
DL	delivery_late	Expected delivery date has passed
DNR	delivered_not_received	Tracking says delivered but customer did not receive it
DP	delivery_promise	A promised or guaranteed delivery commitment was not met
MWI	missing_or_wrong_item	Item is missing, incorrect, or different from ordered
DIP	damaged_item_or_package	Item or package arrived damaged
RR	return_or_refund	Return, refund, cancellation, or related request
PC	payment_or_charge	Payment, charge, billing, or transaction issue
PM	prime_membership	Prime membership-related request
AA	account_or_access	Account, login, or access issue
PDH	product_or_device_help	Product/device usage or troubleshooting
CSF	customer_service_followup	Follow-up on an existing support interaction
OU	other_or_unclear	Unsupported, ambiguous, or unclear request


Delivery-state boundaries
A particularly important distinction is between these delivery-related intents:
- OS: "Where is my order?"
- DL: "The expected delivery date has passed."
- DNR: "Tracking says delivered, but I did not receive it."
- DP: "A promised or guaranteed delivery commitment was not met."
These boundaries were explicitly incorporated into the classification guidelines and evaluation.
8. Golden Evaluation Dataset
The project includes a manually labelled golden set of 200 examples.
The sampling intentionally includes:
- common customer requests
- boundary cases
- short/noisy messages
- multilingual messages
- multi-intent messages
- likely escalation cases
- unclear/other requests
The final split is:
160 examples — development/train split
40 examples  — frozen evaluation split

The labelling methodology and taxonomy definitions are documented in:
golden/labeling_notes.md
golden/annotation_guidelines.md

The evaluation set is also validated for duplicate customer messages before evaluation.
9. Baselines
Classical baselines were established as reference points for the intent-classification task.
Model	Accuracy	Macro F1
Majority class	20.00%	2.56%
Word TF-IDF + Logistic Regression	30.00%	26.51%
Char TF-IDF + Logistic Regression	35.00%	31.79%
LLM classifier	75.00%	74.38%


The classical models provide reference points rather than production alternatives.
The char-level TF-IDF baseline was selected after development cross-validation and then evaluated on the frozen split.
10. Historical Retrieval
Response generation uses TF-IDF retrieval over historical AmazonHelp customer → support response pairs.
To reduce direct evaluation leakage, exact-overlap examples were removed from the retrieval corpus.
Human-reviewed retrieval evaluation
A manually reviewed sample of 50 golden queries was evaluated against the top retrieved historical examples.
Metric	Result
Top-1 relevant	90.0%
Top-1 useful	82.0%
Top-3 relevant	88.7%
Top-3 useful	79.3%


These are human-reviewed retrieval metrics rather than automated semantic-similarity scores.
11. Response Generation
The response-generation pipeline uses the retrieved historical interactions as supporting evidence.
The generation prompt explicitly instructs the model to:
- treat the current customer message as the primary source of truth
- use historical examples as evidence rather than templates to copy blindly
- avoid inventing policies, prices, refunds, credits, delivery dates, or account information
- avoid claiming live access or actions that were not actually performed
- avoid exposing information from other customers
- ask only for information necessary to proceed
- produce a concise customer-facing response
The current generation model is:
ministral-3:3b

12. Escalation Policy
The escalation layer combines classifier confidence, retrieval quality, intent-specific risk, and explicit escalation rules.
A case can be escalated when, for example:
- the intent is other_or_unclear
- classifier confidence is below the configured threshold
- no useful historical example is retrieved
- retrieval similarity is too low
- the request involves financial/security risk
- the case indicates a delivered-but-not-received issue
- the customer indicates a repeated or prolonged unresolved issue
- a refund has been pending for an extended period
The policy is intentionally conservative around cases where automated handling could be unreliable or risky.
13. End-to-End Evaluation
The complete agent is evaluated through:
evaluation/evaluate_agent.py

The evaluation pipeline runs:
Customer message
       ↓
LLM intent classifier
       ↓
TF-IDF historical retrieval
       ↓
LLM response generation
       ↓
Escalation policy
       ↓
Evaluation artifacts

Run:
python -m evaluation.evaluate_agent

Predictions are written to:
evaluation/results/agent_predictions.csv

Metrics can then be calculated with:
python -m evaluation.analysis.evaluate_agent_metrics

14. LLM-as-Judge Evaluation
Generated responses are evaluated using an LLM-as-judge rubric covering:
- relevance
- correctness
- grounding
- helpfulness
- safety
- overall quality
Current results on the 40-example evaluation set:
Dimension	Score
Relevance	2.98 / 5
Correctness	2.58 / 5
Grounding	4.95 / 5
Helpfulness	2.77 / 5
Safety	5.00 / 5
Overall	2.77 / 5


The results are stored in:
evaluation/results/llm_judge_results.csv

The judge should be treated as an evaluation signal rather than independent human ground truth.
No independent second-human annotation was performed for response-generation quality.
15. Failure Analysis
Manual review of the frozen evaluation results identified several recurring failure patterns.
1. Delivery-state boundary confusion
The classifier can confuse:
delivery_late

with:
delivered_not_received

when a message contains mixed delivery-status language.
2. Ambiguous or noisy messages
Short, multilingual, indirect, or conversational messages can contain signals associated with multiple intents without expressing a clear support request.
3. Multi-signal messages
Some customer messages contain both an underlying problem and a requested remedy.
For example:
"My order is 10 days late, can you cancel it and refund me?"

contains both a delivery problem and a requested resolution.
4. Taxonomy coverage limitations
Some real customer messages do not fit naturally into the 13-intent taxonomy and may therefore be mapped to other_or_unclear or a nearby intent.
5. Acknowledgement without resolution
Generated responses can acknowledge the customer's frustration or gratitude without providing a concrete next step.
The generation failure review is stored in:
evaluation/results/generation_failure_review.csv

16. What Is Misleading About My Headline Number?
The 75.0% intent accuracy is useful, but it is not a production-quality estimate.
There are several reasons.
Small evaluation set
The frozen evaluation set contains only 40 examples, so individual examples have a large effect on the headline number.
Development contamination
The frozen evaluation set was consulted during iterative development. It should therefore not be described as a completely untouched holdout.
Accuracy hides intent difficulty
Clear canonical requests are easier than:
- delivery-state boundaries
- multilingual messages
- ambiguous requests
- multi-intent messages
- noisy conversational messages
A single accuracy number does not reveal this distribution of difficulty.
Judge scores are not human truth
The high grounding score from the LLM judge does not prove that generated responses are always correct or useful.
Grounding, correctness, relevance, and helpfulness measure different properties.
The results should therefore be interpreted as development-stage evidence that the pipeline works, rather than as a production-quality performance estimate.
17. Project Structure
hiver-sde-takehome/
│
├── data/
│   └── processed/
│       ├── amazonhelp_conversations.csv
│       ├── amazonhelp_customer_sample.csv
│       └── retrieval_pairs_safe.csv
│
├── evaluation/
│   ├── analysis/
│   │   ├── evaluate_agent_metrics.py
│   │   ├── llm_judge.py
│   │   ├── review_agent_errors.py
│   │   ├── review_agent_responses.py
│   │   └── review_llm_judge.py
│   │
│   ├── golden/
│   ├── results/
│   │   ├── agent_predictions.csv
│   │   ├── generation_failure_review.csv
│   │   └── llm_judge_results.csv
│   │
│   └── evaluate_agent.py
│
├── golden/
│   ├── annotation_guidelines.md
│   ├── labeling_notes.md
│   ├── golden_evaluation.csv
│   ├── evaluation_train.csv
│   └── evaluation_test.csv
│
├── src/
│   ├── escalation/
│   │   └── policy.py
│   │
│   ├── generation/
│   │   ├── config.py
│   │   ├── generator.py
│   │   ├── mock_provider.py
│   │   ├── models.py
│   │   ├── ollama_provider.py
│   │   ├── prompts.py
│   │   ├── provider.py
│   │   └── response_cleaner.py
│   │
│   ├── intents/
│   │   ├── classifier.py
│   │   ├── llm_classifier.py
│   │   ├── models.py
│   │   └── rules.py
│   │
│   ├── pipeline/
│   │   └── agent.py
│   │
│   └── retrieval/
│       └── tfidf_retriever.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore

18. Running the Test Suite
Run the complete automated test suite:
python -m pytest -q

Current test status:
51 passed

The tests cover the classifier contract, LLM classifier behavior, pipeline components, retrieval/generation behavior, and escalation policy.
19. Reproducing the Main Evaluation
Once the dataset and Ollama model are available:
Run the complete agent evaluation
python -m evaluation.evaluate_agent

Calculate agent metrics
python -m evaluation.analysis.evaluate_agent_metrics

Run the LLM-as-judge
python -m evaluation.analysis.llm_judge

Review classifier errors
python -m evaluation.analysis.review_agent_errors

Review generated responses
python -m evaluation.analysis.review_agent_responses

Review worst LLM-judge results
python -m evaluation.analysis.review_llm_judge

The resulting artifacts are stored under:
evaluation/results/

20. Next-Week Plan
If this system were taken forward, I would prioritize:
1. Expand the human evaluation set with a larger, stratified sample.
2. Add an independent second annotator and measure human agreement.
3. Improve delivery-state boundary handling using targeted evaluation cases.
4. Add stronger response-level checks for unsupported claims and unnecessary information requests.
5. Evaluate retrieval using semantic relevance in addition to TF-IDF.
6. Compare multiple local/open models under the same evaluation harness.
7. Introduce confidence calibration and an explicit abstention policy.
8. Evaluate the complete system on a genuinely untouched holdout set.
9. Add production-style observability for classification, retrieval, generation, and escalation decisions.
21. Engineering Notes
Design priorities
The implementation prioritizes:
- modular components
- dependency injection
- deterministic classical baselines
- reproducible evaluation artifacts
- leakage-aware retrieval
- explicit escalation rules
- local model execution
- testable interfaces
- transparent failure analysis
Why TF-IDF retrieval?
For this take-home, TF-IDF provides a lightweight and interpretable retrieval baseline that can be evaluated quickly over a large historical corpus.
A production system would likely compare this against dense embeddings or hybrid retrieval.
Why a local LLM?
Using Ollama keeps the evaluation reproducible without requiring an external API key and makes the complete pipeline runnable locally.
22. Repository
GitHub:
https://github.com/pithiyakeval/hiver-sde-takehome