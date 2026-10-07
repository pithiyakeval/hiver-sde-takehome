from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CaseEventModel(Base):
    """Audit event associated with a support case."""

    __tablename__ = "case_events"

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

    event_type: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
    )

    actor: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    event_metadata: Mapped[dict | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
    )