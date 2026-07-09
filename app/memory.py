"""
memory.py — Session-scoped conversation history, backed by SQLite (via SQLModel + aiosqlite).

Each turn (question + answer) is written after generation and keyed by session_id.
On the next request for the same session_id, the last N turns are read back and
injected into the LLM prompt as prior conversation, so follow-up questions
("what about for a two-wheeler?") resolve against context instead of starting cold.

Environment variables:
    CHAT_DB_PATH — path to the SQLite file (default: ./data/chat_history.db)
"""

import logging
import os
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


class ChatTurn(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    question: str
    # The English-translated form actually used for retrieval on this turn.
    # Reused (prepended) when retrieving for the *next* turn, so a vague
    # follow-up like "what can I challenge this?" inherits the previous
    # turn's topic instead of retrieving cold against just those few words.
    retrieval_question: str = ""
    answer: str
    sector: str | None = None
    detected_lang: str = "en"
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), index=True
    )


_DB_PATH = os.getenv("CHAT_DB_PATH", "./data/chat_history.db")
_engine = create_async_engine(f"sqlite+aiosqlite:///{_DB_PATH}")


async def init_db() -> None:
    """Create the chat_turn table if it doesn't exist, and patch in any
    columns added since. create_all() only creates missing tables — it won't
    alter an existing one — so new columns need a manual ALTER here. No real
    migration tool is set up yet; worth adding Alembic once the schema settles."""
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    async with _engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        result = await conn.exec_driver_sql("PRAGMA table_info(chatturn)")
        existing_columns = {row[1] for row in result.fetchall()}
        if "retrieval_question" not in existing_columns:
            await conn.exec_driver_sql(
                "ALTER TABLE chatturn ADD COLUMN retrieval_question VARCHAR DEFAULT ''"
            )
            logger.info("Migrated chat_history.db: added retrieval_question column")
    logger.info("Chat memory DB ready at %s", _DB_PATH)


async def save_turn(
    session_id: str,
    question: str,
    retrieval_question: str,
    answer: str,
    sector: str | None,
    detected_lang: str,
) -> None:
    """Persist one (question, answer) turn for a session."""
    async with AsyncSession(_engine) as db:
        db.add(
            ChatTurn(
                session_id=session_id,
                question=question,
                retrieval_question=retrieval_question,
                answer=answer,
                sector=sector,
                detected_lang=detected_lang,
            )
        )
        await db.commit()


async def get_recent_turns(session_id: str, limit: int = 4) -> list[ChatTurn]:
    """Return the last `limit` turns for a session, oldest first (for prompt order)."""
    async with AsyncSession(_engine) as db:
        stmt = (
            select(ChatTurn)
            .where(ChatTurn.session_id == session_id)
            .order_by(ChatTurn.created_at.desc())
            .limit(limit)
        )
        result = await db.exec(stmt)
        return list(reversed(result.all()))
