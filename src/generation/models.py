from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class RetrievedExample:
    customer_text:str
    historical_response:str
    similarity:float

@dataclass(frozen=True)
class GenerationInput:
    customer_message:str
    intent:str
    retrieved_examples:List[RetrievedExample]

@dataclass(frozen=True)
class GenerationResult:
    draft_response:str
    model:str
    grounded:bool
