from prometheus_client import Counter, Histogram


# ==============================
# HTTP metrics
# ==============================

HTTP_REQUESTS_TOTAL = Counter(
    "support_http_requests_total",
    "Total number of HTTP requests.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "support_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)


# ==============================
# Support analysis metrics
# ==============================

SUPPORT_ANALYSES_TOTAL = Counter(
    "support_analyses_total",
    "Total number of support analyses.",
)

SUPPORT_ANALYSES_FAILED_TOTAL = Counter(
    "support_analyses_failed_total",
    "Total number of failed support analyses.",
)

SUPPORT_ANALYSIS_DURATION = Histogram(
    "support_analysis_duration_seconds",
    "Support analysis duration in seconds.",
)


# ==============================
# Pipeline stage metrics
# ==============================

CLASSIFICATION_DURATION = Histogram(
    "support_classification_duration_seconds",
    "Intent classification duration in seconds.",
)

RETRIEVAL_DURATION = Histogram(
    "support_retrieval_duration_seconds",
    "Historical retrieval duration in seconds.",
)

GENERATION_DURATION = Histogram(
    "support_generation_duration_seconds",
    "Response generation duration in seconds.",
)

ESCALATION_DURATION = Histogram(
    "support_escalation_duration_seconds",
    "Escalation policy evaluation duration in seconds.",
)


# ==============================
# Support quality / routing metrics
# ==============================

SUPPORT_ESCALATIONS_TOTAL = Counter(
    "support_escalations_total",
    "Total number of support cases escalated.",
)

SUPPORT_LOW_CONFIDENCE_TOTAL = Counter(
    "support_low_confidence_total",
    "Total number of analyses with low intent confidence.",
)

SUPPORT_LLM_ERRORS_TOTAL = Counter(
    "support_llm_errors_total",
    "Total number of LLM-related errors.",
)