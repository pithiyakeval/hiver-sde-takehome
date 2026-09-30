from pathlib import Path


SEED = 42

SAMPLE_SIZE = 200

CANDIDATE_POOL_MULTIPLIER = 4

MAX_BUCKET_CANDIDATES = 350

SAMPLING_TARGETS = {
    "common_normal": 100,
    "boundary_case": 40,
    "short_noisy": 20,
    "multilingual": 15,
    "multi_intent": 10,
    "likely_escalation": 10,
    "other_unclear_candidate": 5,
}

OUTPUT_DIR = Path("golden")

INPUT_PATH = Path(
    "data/processed/amazonhelp_conversations.csv"
)