from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CaseModel(Base):
    """Persistent representation of a support case."""

    __tablename__ = "support_cases"

    __table_args__ = (
        CheckConstraint(
            "intent_confidence >= 0.0 AND intent_confidence <= 1.0",
            name="ck_support_cases_intent_confidence",
        ),
    )

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    case_id: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
        nullable=False,
    )

    # ---------------------------------------------------------
    # Customer request
    # ---------------------------------------------------------

    customer_message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        index=True,
        nullable=False,
    )

    intent: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    intent_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    classification_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # ---------------------------------------------------------
    # Generated response
    # ---------------------------------------------------------

    draft_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    generation_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    grounded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    # ---------------------------------------------------------
    # Escalation
    # ---------------------------------------------------------

    should_escalate: Mapped[bool] = mapped_column(
        Boolean,
        index=True,
        nullable=False,
    )

    escalation_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    escalation_reason_code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    # ---------------------------------------------------------
    # AI pipeline performance
    # ---------------------------------------------------------

    classification_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    retrieval_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    generation_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    escalation_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    total_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # ---------------------------------------------------------
    # Timestamps
    # ---------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )