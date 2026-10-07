from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db():
    """Provide one SQLAlchemy session per request."""

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()