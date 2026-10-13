"""
질문 기록 라우터
─────────────────────────────────────────────────────────────
  1. GET  /history : 로그인한 사용자의 최근 질문 기록 (최신순, 최대 50개)
  2. POST /history  : 질문 기록 추가

둘 다 로그인이 필요합니다 (get_current_user 의존성).
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..dependencies import get_current_user
from ..models import HistoryEntry, User

router = APIRouter()


class HistoryEntryIn(BaseModel):
    question: str
    age: Optional[int] = None


class HistoryEntryOut(BaseModel):
    question: str
    age: Optional[int]
    created_at: datetime


@router.get("", response_model=List[HistoryEntryOut])
async def list_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(HistoryEntry)
        .where(HistoryEntry.user_id == user.id)
        .order_by(HistoryEntry.id.desc())
        .limit(50)
    )
    return result.scalars().all()


@router.post("", response_model=HistoryEntryOut, status_code=201)
async def add_history(
    body: HistoryEntryIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    question = body.question.strip()
    entry = HistoryEntry(user_id=user.id, question=question, age=body.age)
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry
