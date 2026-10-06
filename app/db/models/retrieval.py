from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text

class CaseRetrievalModel(Base):
    """Historical retrieval evidence associated with a support case."""

    __tablename__ = "case_retrievals"

    __table_args__ = (
        CheckConstraint(
            "similarity >= 0.0 AND similarity <= 1.0",
            name="ck_case_retrievals_similarity",
        ),
        CheckConstraint(
            "rank >= 1",
            name="ck_case_retrievals_rank",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    case_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("support_cases.case_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    customer_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    historical_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    similarity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )