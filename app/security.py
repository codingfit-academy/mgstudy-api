"""
비밀번호 해싱 / 토큰 생성
─────────────────────────────────────────────────────────────
bcrypt는 72바이트를 넘는 입력을 자르기 때문에, 그 이상은 애초에
막아서(Pydantic 스키마의 max_length) 잘림으로 인한 혼동을 피한다.
"""
import secrets

import bcrypt

SESSION_TTL_DAYS = 30


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def generate_token() -> str:
    return secrets.token_urlsafe(32)
