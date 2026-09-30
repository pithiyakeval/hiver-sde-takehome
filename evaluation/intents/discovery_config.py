from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_discovery"
)

SAMPLE_PATH = (
    OUTPUT_DIR
    / "intent_discovery_sample.csv"
)

PHRASES_PATH = (
    OUTPUT_DIR
    / "top_phrases.csv"
)

TOPICS_PATH = (
    OUTPUT_DIR
    / "topic_summary.csv"
)

EXAMPLES_PATH = (
    OUTPUT_DIR
    / "topic_examples.csv"
)

REVIEW_PATH = (
    OUTPUT_DIR
    / "intent_review.csv"
)


# Reproducibility / discovery settings

RANDOM_STATE = 42

# Number of unique customer messages used for discovery.
DISCOVERY_SAMPLE_SIZE = 5_000

# Candidate themes.
#
# IMPORTANT:
# These are NOT the final intents.
# They are exploratory topics that we will inspect.
N_TOPICS = 12

# Number of representative examples per topic.
EXAMPLES_PER_TOPIC = 20

# Number of phrases shown for each topic.
TOP_TERMS_PER_TOPIC = 15

# TF-IDF vocabulary limits.
MIN_DOCUMENT_FREQUENCY = 5
MAX_DOCUMENT_FREQUENCY = 0.85
MAX_FEATURES = 20_000


# Text-discovery configuration

CUSTOM_STOPWORDS = {
    "amazon",
    "amazonhelp",
    "help",
    "please",
    "thanks",
    "thank",
    "hi",
    "hello",
    "hey",
    "customer",
    "service",
    "support",
    "team",
    "guys",
    "need",
    "want",
    "just",
    "like",
    "really",
    "today",
    "now",
    "know",
    "got",
    "get",
    "getting",
    "did",
    "does",
    "dont",
    "didnt",
    "cant",
    "could",
    "would",
    "im",
    "ive",
    "id",
    "youre",
    "weve",
    "amp",
    "https",
    "http",
    "www",
}


CONTRACTION_REPLACEMENTS = {
    "can't": "cannot",
    "won't": "willnot",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "we're": "we are",
    "we've": "we have",
    "they're": "they are",
}