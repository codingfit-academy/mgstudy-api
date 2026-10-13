"""
공통 의존성
─────────────────────────────────────────────────────────────
get_current_user: Authorization: Bearer <token> 헤더를 sessions 테이블에서
조회해 로그인한 사용자를 반환합니다. 토큰이 없거나 만료됐으면 401.
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db
from .models import Session, User


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="로그인이 필요해요")
    token = authorization.removeprefix("Bearer ").strip()

    result = await db.execute(select(Session).where(Session.token == token))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=401, detail="로그인이 만료됐어요. 다시 로그인해주세요")

    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        # SQLite 등 일부 드라이버는 tz 정보 없이 반환하므로 UTC로 간주한다.
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="로그인이 만료됐어요. 다시 로그인해주세요")

    user = await db.get(User, session.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없어요")
    return user
