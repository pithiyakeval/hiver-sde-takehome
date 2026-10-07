# Hiver SDE Take-Home — AI Support Agent

Evaluation-driven AI customer-support agent built on the **Customer Support on Twitter** dataset.

The system:

- classifies customer messages into support intents
- retrieves similar historical AmazonHelp interactions
- drafts a grounded customer-facing response
- decides whether to auto-handle or escalate
- persists cases and audit history through a FastAPI service

Built with **Python, FastAPI, Ollama, Ministral 3B, TF-IDF, SQLAlchemy and SQLite**.

---

## What it does

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
Response Generation
       │
       ▼
Risk-aware Escalation
       │
       ▼
Support Case + Audit History
```

The project uses **AmazonHelp** as the target support account.

Dataset subset:

| | Approx. |
|---|---:|
| AmazonHelp support tweets | 169K |
| Customer → AmazonHelp response pairs | 168K |
| Unique customer authors | 71K |

---

## Results

The current development-stage evaluation uses a manually labelled 200-example
golden set with a 160-example development split and a 40-example evaluation
split.

### Intent Classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority class | 20.0% | 2.56% |
| Word TF-IDF + Logistic Regression | 30.0% | 26.51% |
| Char TF-IDF + Logistic Regression | 35.0% | 31.79% |
| Hybrid rules + Char TF-IDF | 37.5% | 36.77% |
| Standalone LLM classifier | 62.5% | 62.84% |

### End-to-End Agent

| Metric | Result |
|---|---:|
| Intent accuracy | **75.0%** |
| Intent macro F1 | **74.38%** |
| Escalation accuracy | **65.0%** |
| Top-1 retrieval relevant | **90.0%** |
| Top-1 retrieval useful | **82.0%** |
| LLM-judge grounding | **4.95 / 5** |
| LLM-judge safety | **5.00 / 5** |

The standalone LLM result measures classification in isolation. The 75.0%
result measures the complete agent's final intent output after the production
classification pipeline and deterministic decision logic.

> **Evaluation caveat:** the 40-example evaluation split was consulted during
> iterative development. It is therefore development-stage evidence, not an
> untouched final holdout.

---

## Architecture

The project is intentionally split into an AI pipeline and a production-style
API layer.

### AI pipeline

```text
src/
├── intents/          # Intent classification
├── retrieval/       # Historical interaction retrieval
├── generation/      # Grounded response generation
├── escalation/      # Auto-handle / escalation policy
└── pipeline/        # End-to-end orchestration
```

### API and persistence

```text
app/
├── api/             # HTTP routes and error handling
├── services/        # Application/business logic
├── repositories/    # Persistence abstractions
├── models/          # Domain models
├── schemas/         # API schemas
└── db/              # SQLAlchemy database layer
```

The API uses:

- FastAPI
- SQLAlchemy
- SQLite
- dependency injection
- repository interfaces
- Unit of Work transactions
- explicit case-state transitions
- persistent audit events

### Case lifecycle

```text
READY_FOR_REVIEW
       │
       ├── approve ──► AUTO_HANDLED ──► RESOLVED
       │
       └── escalate ─► ESCALATED ─────► RESOLVED
```

Invalid state transitions return a structured `409` response.

Important transitions create audit events:

```text
CASE_CREATED
CASE_APPROVED
CASE_ESCALATED
CASE_RESOLVED
```

---

## Evaluation

The evaluation harness combines:

- human-labelled intent examples
- classical ML baselines
- standalone LLM classification
- end-to-end agent evaluation
- human-reviewed retrieval evaluation
- LLM-as-judge response evaluation
- manual failure analysis

### Golden set

The 200 labelled examples intentionally cover:

- common requests
- delivery boundary cases
- noisy and short messages
- multilingual messages
- multi-intent requests
- likely escalation cases
- unclear requests

The final split is:

```text
160 development examples
40 evaluation examples
```

### Retrieval

Historical AmazonHelp customer → support pairs are indexed using TF-IDF.

Exact-overlap examples were removed from the retrieval corpus to reduce direct
evaluation leakage.

Human review over 50 golden queries:

| Metric | Result |
|---|---:|
| Top-1 relevant | 90.0% |
| Top-1 useful | 82.0% |
| Top-3 relevant | 88.7% |
| Top-3 useful | 79.3% |

### LLM-as-judge

Generated responses were evaluated for relevance, correctness, grounding,
helpfulness and safety.

| Dimension | Score |
|---|---:|
| Relevance | 2.98 / 5 |
| Correctness | 2.58 / 5 |
| Grounding | 4.95 / 5 |
| Helpfulness | 2.78 / 5 |
| Safety | 5.00 / 5 |
| Overall | 2.78 / 5 |

LLM-as-judge scores are treated as evaluation signals rather than independent
human ground truth.

---

## Intent taxonomy

The final taxonomy contains 13 intents:

| Code | Intent | Description |
|---|---|---|
| OS | `order_status` | Current order status/location |
| DL | `delivery_late` | Expected delivery date has passed |
| DNR | `delivered_not_received` | Tracking says delivered but customer did not receive it |
| DP | `delivery_promise` | Promised/guaranteed delivery was not met |
| MWI | `missing_or_wrong_item` | Missing, incorrect, or wrong item |
| DIP | `damaged_item_or_package` | Damaged item/package |
| RR | `return_or_refund` | Return, refund or cancellation |
| PC | `payment_or_charge` | Payment/billing/transaction issue |
| PM | `prime_membership` | Prime membership issue |
| AA | `account_or_access` | Account/login/access issue |
| PDH | `product_or_device_help` | Product/device help |
| CSF | `customer_service_followup` | Follow-up on existing support |
| OU | `other_or_unclear` | Ambiguous or unsupported request |

The most important delivery boundaries are:

```text
OS  → "Where is my order?"

DL  → "The expected delivery date has passed."

DNR → "Tracking says delivered, but I did not receive it."

DP  → "A promised or guaranteed delivery commitment was not met."
```

These boundaries were explicitly included in the annotation guidelines because
they are a recurring source of classifier errors.

---

## Quickstart

### Prerequisites

- Python 3.x
- Git
- Ollama
- `ministral-3:3b`

### Clone

```bash
git clone https://github.com/pithiyakeval/hiver-sde-takehome.git
cd hiver-sde-takehome
```

### Create environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dataset

Download the Kaggle **Customer Support on Twitter** dataset and place:

```text
twcs.csv
```

in the project root.

### Ollama

```bash
ollama pull ministral-3:3b
```

Verify:

```bash
ollama run ministral-3:3b "Say OK"
```

The application uses:

```text
http://127.0.0.1:11434
```

No external LLM API key is required.

---

## Run the API

```bash
uvicorn app.main:app --reload
```

API base:

```text
http://127.0.0.1:8000/api/v1
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/support/analyze` | Analyze a customer message |
| `GET` | `/support/cases` | List support cases |
| `GET` | `/support/cases/{case_id}` | Get a case |
| `GET` | `/support/cases/{case_id}/events` | Get audit events |
| `POST` | `/support/cases/{case_id}/approve` | Approve automated handling |
| `POST` | `/support/cases/{case_id}/escalate` | Escalate to human |
| `POST` | `/support/cases/{case_id}/resolve` | Resolve a case |

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/support/analyze \
  -H "Content-Type: application/json" \
  -d "{\"customer_message\":\"My package says delivered but I never received it.\"}"
```

The response includes:

- predicted intent
- confidence
- classification reason
- retrieved historical examples
- generated draft
- grounding status
- escalation decision
- performance timings
- persistent case ID

---

## Reproduce the evaluation

Run the complete agent evaluation:

```bash
python -m evaluation.evaluate_agent
```

Calculate metrics:

```bash
python -m evaluation.analysis.evaluate_agent_metrics
```

Run the LLM judge:

```bash
python -m evaluation.analysis.llm_judge
```

Review classifier errors:

```bash
python -m evaluation.analysis.review_agent_errors
```

Review generated responses:

```bash
python -m evaluation.analysis.review_agent_responses
```

Review worst LLM-judge results:

```bash
python -m evaluation.analysis.review_llm_judge
```

Evaluation artifacts are stored under:

```text
evaluation/results/
```

---

## Engineering decisions

### Why TF-IDF?

TF-IDF provides a lightweight, deterministic and interpretable retrieval
baseline that can be evaluated efficiently over a large historical corpus.

A production system would compare it with dense or hybrid retrieval.

### Why a local LLM?

Ollama makes the complete pipeline reproducible locally without requiring an
external API key.

### Why conservative escalation?

Automation should fail safely.

The policy escalates cases when signals indicate that automated handling may be
unreliable or risky, including:

- unclear intent
- low classifier confidence
- weak retrieval
- financial/security-sensitive requests
- delivered-but-not-received cases
- prolonged unresolved issues

### Why Unit of Work?

Case state changes and audit events should commit atomically.

```text
status update
      +
audit event
      ↓
single transaction
      ↓
    COMMIT
```

On failure:

```text
failure
   ↓
ROLLBACK
```

This keeps persisted case state consistent with its audit history.

---

## Failure analysis

Manual review of the evaluation errors identified five recurring patterns:

1. **Delivery-state boundary confusion**
   `delivery_late` vs `delivered_not_received`

2. **Underlying issue vs requested remedy**
   A late order may also contain a refund/cancellation request.

3. **Noisy or multilingual language**
   Short messages, spelling errors, code-switching and informal syntax.

4. **Taxonomy coverage**
   Some real requests do not fit naturally into the 13 supported intents.

5. **Acknowledgement without resolution**
   Historical support responses can contain acknowledgement without a concrete
   next step, which generation can inherit.

The detailed failure artifacts are stored under:

```text
evaluation/results/
```

---

## Limitations & next steps

The current results are development-stage evidence.

The main limitations are:

- small 40-example evaluation split
- evaluation examples were consulted during development
- LLM-as-judge is not independent human ground truth
- no independent second-human annotation for retrieval/generation quality
- TF-IDF retrieval is a lightweight baseline
- confidence calibration is not yet production-grade

If taken further, the highest-value improvements would be:

1. larger stratified human evaluation set
2. independent second annotator and agreement measurement
3. stronger delivery-state boundary handling
4. response checks for unsupported claims
5. dense/hybrid retrieval comparison
6. evaluation across multiple open/local models
7. calibrated confidence and stronger abstention
8. genuinely untouched holdout evaluation
9. production quality/latency/escalation monitoring

---

## Project structure

```text
hiver-sde-takehome/
│
├── app/                    # FastAPI + persistence
├── src/                    # AI agent pipeline
├── evaluation/             # Evaluation harness
├── golden/                 # Human-labelled evaluation data
├── data/                   # Processed dataset artifacts
├── tests/                  # Automated tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Repository

**GitHub:**
https://github.com/pithiyakeval/hiver-sde-takehome

---

## Summary

This project focuses on **measurability, safe automation, reproducibility and
failure analysis** rather than treating response generation as the only
objective.

```text
Intent Classification
        ↓
Historical Retrieval
        ↓
Grounded Generation
        ↓
Risk-aware Escalation
        ↓
Case Workflow
        ↓
Audit History
        ↓
Evaluation
```