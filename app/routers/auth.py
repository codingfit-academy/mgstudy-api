"""
인증 라우터 (회원가입 / 로그인 / 로그아웃 / 내 정보)
─────────────────────────────────────────────────────────────
  1. POST  /auth/signup  : 회원가입 (bcrypt 해시 저장) + 세션 발급
  2. POST  /auth/login   : 로그인 + 세션 발급
  3. POST  /auth/logout  : 현재 세션 폐기
  4. GET   /auth/me      : 내 정보 조회
  5. PATCH /auth/me      : 닉네임/아바타 수정

토큰은 JWT가 아니라 DB `sessions` 테이블의 opaque 토큰이다. 프론트가
`Authorization: Bearer <token>` 헤더로 보내고, 로그아웃 시 해당 세션 행을
지우면 즉시 무효화된다 (JWT 블록리스트가 필요 없음).
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..dependencies import get_current_user
from ..models import Session, User
from ..security import SESSION_TTL_DAYS, generate_token, hash_password, verify_password

router = APIRouter()


def _validate_email(value: str) -> str:
    value = value.strip()
    if not value or "@" not in value:
        raise ValueError("올바른 이메일 형식이 아니에요")
    return value


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    # 선택. AI가 낱말·문장 난이도를 맞추는 데만 쓴다 (비우면 초등학생 기본 수준).
    age: Optional[int] = Field(default=None, ge=1, le=19)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("이름을 입력해주세요")
        return v

    @field_validator("email")
    @classmethod
    def _email_valid(cls, v: str) -> str:
        return _validate_email(v)

    @field_validator("password")
    @classmethod
    def _password_len(cls, v: str) -> str:
        if len(v) < 4 or len(v) > 72:
            raise ValueError("비밀번호는 4자 이상 72자 이하로 입력해주세요")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _email_valid(cls, v: str) -> str:
        return _validate_email(v)


class UpdateProfileRequest(BaseModel):
    nickname: Optional[str] = None
    avatar: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=1, le=19)


class AccountOut(BaseModel):
    name: str
    email: str
    nickname: str
    avatar: str
    age: Optional[int]


class AuthResult(BaseModel):
    token: str
    account: AccountOut


def _to_account_out(user: User) -> AccountOut:
    return AccountOut(
        name=user.name,
        email=user.email,
        nickname=user.nickname,
        avatar=user.avatar,
        age=user.age,
    )


async def _create_session(db: AsyncSession, user_id: int) -> Session:
    session = Session(
        user_id=user_id,
        token=generate_token(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.post("/signup", response_model=AuthResult, status_code=201)
async def signup(body: SignupRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이미 가입된 이메일이에요")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
        nickname=body.name,
        avatar="🦕",
        age=body.age,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    session = await _create_session(db, user.id)
    return AuthResult(token=session.token, account=_to_account_out(user))


@router.post("/login", response_model=AuthResult)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않아요")

    session = await _create_session(db, user.id)
    return AuthResult(token=session.token, account=_to_account_out(user))


@router.post("/logout", status_code=204)
async def logout(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        await db.execute(delete(Session).where(Session.token == token))
        await db.commit()


@router.get("/me", response_model=AccountOut)
async def me(user: User = Depends(get_current_user)):
    return _to_account_out(user)


@router.patch("/me", response_model=AccountOut)
async def update_me(
    body: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.nickname is not None:
        nickname = body.nickname.strip()
        user.nickname = nickname if nickname else user.name
    if body.avatar is not None:
        user.avatar = body.avatar
    if body.age is not None:
        user.age = body.age
    await db.commit()
    await db.refresh(user)
    return _to_account_out(user)
