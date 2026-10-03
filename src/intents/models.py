from dataclasses import dataclass


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    reason: str