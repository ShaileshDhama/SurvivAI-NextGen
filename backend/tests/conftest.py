"""
Test configuration for FastAPI application tests
"""

import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.main import app
from app.db.base import Base, get_db
from app.core.config import settings

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_survivai"

# Create test engine
engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=NullPool,
)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db():
    """
    Create a fresh database on each test case.
    """
    # Create the database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Run the tests
    async with TestingSessionLocal() as session:
        yield session

    # Drop the database after the tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Override get_db dependency
@pytest.fixture
async def override_get_db(db):
    """
    Override get_db dependency for testing.
    """
    async def _override_get_db():
        try:
            yield db
        finally:
            pass
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_get_db) -> Generator:
    """
    Create a new FastAPI TestClient that uses the `override_get_db` fixture.
    """
    with TestClient(app) as client:
        yield client
