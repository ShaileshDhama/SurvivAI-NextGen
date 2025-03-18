"""
Database setup and configuration with SQLAlchemy 2.0 async pattern
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

# Create async engine for PostgreSQL
engine = create_async_engine(settings.DATABASE_URI, echo=True)

# Session factory for async sessions
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for all models
Base = declarative_base()

# Dependency to get DB session
async def get_db():
    """
    Dependency for getting async DB session
    Use with FastAPI's Depends
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
