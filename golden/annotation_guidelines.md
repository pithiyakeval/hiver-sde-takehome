# Golden Set Annotation Guidelines

## Purpose

These guidelines define how customer-support messages are assigned to the
13-intent taxonomy used for evaluation of the AmazonHelp support agent.

The goal is to produce consistent labels across annotators while preserving
the ambiguity present in real customer-support messages.

---

## 1. General Annotation Principles

### 1.1 Label the customer's underlying problem

Choose the intent that best describes the **underlying customer problem**.

Do not choose an intent solely because a keyword appears in the message.

For example:

- A customer mentions a payment but the actual problem is that a Kindle
  book did not download → `product_or_device_help`
- A customer asks for a refund because the wrong item was delivered →
  `missing_or_wrong_item`
- A customer mentions a refund while primarily reporting an order problem →
  label the underlying order problem when it is clear.

### 1.2 Requested remedy is secondary

Words such as:

- refund
- cancel
- return
- replacement
- contact support

may describe what the customer wants Amazon to do rather than what caused
the support issue.

When both are present, prefer the underlying problem when it is clearly
identifiable.

### 1.3 Use the message as the primary evidence

Base the label on the customer's message itself.

Do not assume:

- current Amazon policies
- account state
- delivery status not stated in the message
- whether a refund was actually issued
- whether an order was actually cancelled
- information from external links

### 1.4 Do not infer a specific intent from an isolated keyword

Examples:

- "Prime" does not automatically mean `prime_membership`.
- "payment" does not automatically mean `payment_or_charge`.
- "return" does not automatically mean `return_or_refund`.
- "delivered" does not automatically mean `delivered_not_received`.

The surrounding customer request determines the intent.

---

# 2. Primary-Intent Rule for Multi-Intent Messages

Some customer messages contain multiple problems.

When multiple intents are present:

1. Identify the customer's main support problem.
2. Prefer the issue that most directly explains why the customer needs
   support.
3. Treat remedies, consequences, and incidental details as secondary.
4. If no single supported intent clearly dominates, use
   `other_or_unclear`.

### Example

> "Wrong item delivered. I requested a return but nobody responded."

Primary problem:

`missing_or_wrong_item`

The return request is the remedy/follow-up.

---

### Another example

> "Payment was taken but my Kindle book did not download."

Primary problem:

`product_or_device_help`

The payment is supporting context rather than the main product issue.

---

### Another example

> "My cancelled order was charged and I haven't received the refund."

Primary problem:

`payment_or_charge`

The charge is the concrete financial problem; the missing refund is the
requested resolution/consequence.

---

# 3. Delivery Intent Boundaries

Delivery-related intents are intentionally separated because they represent
different customer states.

Use the following decision order.

## 3.1 `delivered_not_received` (DNR)

Use when:

> Tracking or the delivery system indicates that the package was delivered,
> but the customer says they did not receive it.

Strong indicators:

- "shows delivered"
- "marked delivered"
- "tracking says delivered"
- "delivered but I don't have it"
- "package was supposedly delivered"

Do not use DNR merely because the customer says:

> "It wasn't delivered."

There must be evidence or wording indicating a **delivered status**.

---

## 3.2 `delivery_late` (DL)

Use when:

> The expected delivery date has passed or the order is clearly delayed.

Examples:

- "It's 10 days late."
- "Expected yesterday but still not here."
- "Why is my order so late?"
- "It was supposed to arrive three days ago."

The defining feature is a **missed or overdue delivery expectation**.

---

## 3.3 `delivery_promise` (DP)

Use when:

> A specific promised, guaranteed, same-day, next-day, or Prime delivery
> service is the focus of the complaint.

Examples:

- "I paid for next-day delivery and it didn't arrive."
- "Prime promised delivery today."
- "Why didn't my same-day delivery arrive?"
- "What's the point of Prime delivery if the promised date isn't met?"

The defining feature is a **specific delivery promise/service commitment**.

---

## 3.4 `order_status` (OS)

Use when:

> The customer is primarily asking about the current status or location
> of an order, without a clearly established late, delivered-not-received,
> or specific delivery-promise failure.

Examples:

- "Where is my order?"
- "Can you tell me the status?"
- "Why hasn't my order moved?"
- "What is happening with the rest of my order?"

---

## Delivery Decision Summary

Use this hierarchy:

```text
Tracking says delivered + customer did not receive
        → DNR

Expected/promised date passed
        → DL

Specific promised/guaranteed/Prime delivery failed
        → DP

General order/status/location question
        → OS

If the message contains multiple delivery signals, choose the most specific
supported state based on the evidence explicitly present in the message.
4. Intent Definitions
order_status (OS)
General order status, location, progress, or shipment-state question.
Use when the customer wants to know what is happening with an order and no
more specific delivery failure is established.
delivery_late (DL)
An order is late or missed its expected delivery date.
delivered_not_received (DNR)
Tracking/status says delivered, but the customer did not receive the package.
delivery_promise (DP)
A specific delivery promise or service commitment failed, such as Prime,
same-day, next-day, or guaranteed delivery.
missing_or_wrong_item (MWI)
The customer received the wrong item, an item is missing from an order, or
an expected component/item was not included.
Focus on the physical order-content problem.
damaged_item_or_package (DIP)
The product or package arrived damaged, broken, crushed, opened, or otherwise
physically defective.
return_or_refund (RR)
The primary customer problem is a return or refund request/process that is
not better explained by another underlying issue.
If a specific underlying issue clearly explains the request, prefer that
underlying intent.
payment_or_charge (PC)
Payment, charge, billing, unauthorized charge, transaction, or financial
account issue is the primary problem.
Do not choose PC merely because payment is mentioned as background context.
prime_membership (PM)
The customer is primarily asking about or having trouble with Prime
membership, subscription, trial, renewal, or membership benefits.
A simple mention of Prime does not automatically qualify.
account_or_access (AA)
The customer has an account, login, access, identity, or account-management
problem.
product_or_device_help (PDH)
The customer needs help with a product, device, digital content, product
functionality, product page, or product-specific behavior.
This can include Kindle/product-content issues even when payment is also
mentioned, if the product problem is the underlying issue.
customer_service_followup (CSF)
The customer is primarily following up on an existing support interaction,
complaint, replacement, unresolved case, or repeated contact with support.
Typical signals:
- "I've contacted you multiple times."
- "Nobody has responded."
- "I've been chasing this for weeks."
- "I already spoke to support."
- "This is terrible customer service."
Use CSF when the support interaction itself is the primary unresolved
problem.
If a concrete underlying problem clearly dominates, prefer that underlying
intent.
other_or_unclear (OU)
Use when:
- the message does not describe a supported customer-support problem,
- the intent is too ambiguous to assign reliably,
- the message is primarily commentary/social content,
- the message is too noisy to determine the customer's actual request,
- or multiple competing intents exist without a clear primary intent.
Do not use OU simply because the message is short or informal.
5. Noisy, Informal, and Multilingual Messages
Customer messages may contain:
- spelling mistakes
- abbreviations
- slang
- missing punctuation
- URLs
- usernames
- hashtags
- multiple languages
- OCR-like text
- emotional language
Annotators should focus on the semantic customer request rather than writing
quality.
For multilingual messages, infer the intent when the meaning is sufficiently
clear.
If the meaning cannot be determined reliably, use other_or_unclear.
6. Confidence
Annotators record confidence independently of the intent label.
Use:
High
The intent is directly supported by the message and there is little
reasonable ambiguity.
Medium
The intent is plausible but the message contains indirect wording,
multiple signals, or some uncertainty.
Low
The message is highly ambiguous, noisy, incomplete, or difficult to map to
the taxonomy.
7. Ambiguity Flag
Set Ambiguous = yes when:
- two or more intents are genuinely plausible,
- the primary intent depends on interpretation,
- the message contains competing support problems,
- or the message is too noisy/indirect for a confident single interpretation.
Do not mark every difficult example as ambiguous. Use the flag when another
reasonable annotator could plausibly select a different supported intent.
8. Important Boundary Rules
Underlying issue vs remedy
Prefer:
wrong item → MWI
damaged item → DIP

over:
return requested → RR
replacement requested → RR

when the underlying issue is explicit.
Product problem vs payment context
If:
"I paid for the Kindle book but it won't download."

prefer:
PDH

because the product/content problem is the underlying issue.
If:
"I was charged twice for the same order."

prefer:
PC

because the financial problem is primary.
Prime mention vs Prime membership
A message mentioning Prime is not automatically PM.
Use PM when membership, trial, subscription, renewal, or membership
benefits are themselves the support problem.
If Prime is only being mentioned as context for a delivery complaint, use
the appropriate delivery intent.
Support follow-up vs underlying issue
Use CSF when the main problem is an unresolved support interaction.
If the customer says:
"I've contacted you three times because my package is still late."

and the delivery delay is clearly the main issue, delivery_late may be more
appropriate.
If the customer says:
"I've contacted support repeatedly and nobody will respond."

use customer_service_followup.
9. Escalation Labels
Escalation is annotated separately from intent.
Do not change the intent simply because a case should be escalated.
A case can have:
Intent = payment_or_charge
Escalate = yes

or:
Intent = delivery_late
Escalate = no

The escalation decision should be based on the separate escalation criteria,
not on the intent label alone.
10. Annotation Procedure
For every message:
1. Read the complete customer message.
2. Identify the customer's actual problem/request.
3. Identify whether there are multiple competing issues.
4. Apply the primary-intent rule.
5. Apply the delivery-specific boundaries if applicable.
6. Choose exactly one intent.
7. Record confidence.
8. Mark ambiguity when another reasonable intent is plausible.
9. Record a short evidence/reasoning note when useful.
10. Do not use external information to resolve missing context.
11. Handling Genuine Ambiguity
The objective is consistent annotation, not forcing artificial certainty.
If two intents are both genuinely plausible and the message does not provide
enough evidence to choose between them:
Choose the best-supported primary intent
+
Set Ambiguous = yes
+
Explain the ambiguity briefly.

If no supported intent can reasonably be selected:
other_or_unclear

This preserves difficult real-world examples instead of hiding them.
12. Evaluation Principle
The golden set should remain frozen after annotation.
Do not change gold labels simply to improve model accuracy or human agreement.
If an annotation guideline is improved after agreement analysis, document the
change and use the updated guideline for future annotation rounds.
Existing evaluation results should remain traceable to the labels that were
actually used to produce them.

### Why this version is better

The important additions are:

1. **Underlying issue vs remedy**
2. **Explicit multi-intent primary-intent rule**
3. **Much sharper OS / DL / DNR / DP boundaries**
4. **Prime mention ≠ Prime membership**
5. **Product problem vs payment context**
6. **CSF vs underlying issue**
7. **Don't use model performance to alter the gold set**
8. **Ambiguity is explicitly preserved instead of hidden**

This directly addresses what we discovered from the 22 disagreements without artificially improving the κ.

### After saving the file

Run:

```powershell
git diff --check