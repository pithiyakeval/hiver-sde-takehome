from app.models.case import SupportCase
from app.repositories.case_repository import CaseRepository


class InMemoryCaseRepository(CaseRepository):
    """In-memory implementation for local/demo execution."""

    def __init__(self) -> None:
        self._cases: dict[str, SupportCase] = {}

    def save(self, case: SupportCase) -> SupportCase:
        self._cases[case.case_id] = case
        return case

    def get(self, case_id: str) -> SupportCase | None:
        return self._cases.get(case_id)

    def list(self) -> list[SupportCase]:
        return list(self._cases.values())

    def update(self, case: SupportCase) -> SupportCase:
        if case.case_id not in self._cases:
            raise KeyError(f"Case not found: {case.case_id}")

        self._cases[case.case_id] = case
        return case