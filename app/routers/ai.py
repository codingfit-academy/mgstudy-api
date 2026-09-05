"""
AI 라우터 (질문 검증 / 육하원칙 종합)
─────────────────────────────────────────────────────────────
Claude API를 이용해 두 가지를 처리합니다.
  1. POST /ai/check-question   : 아이가 쓴 문장이 진짜 질문인지 확인
  2. POST /ai/compose-question : 프레임워크(예: 육하원칙) 답변 조각들을
                                  자연스러운 한 문장 질문으로 종합

ANTHROPIC_API_KEY 는 app/config.py 의 settings 에서만 읽습니다.
"""
from typing import List, Optional

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings

router = APIRouter()

_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="AI 기능이 아직 설정되지 않았어요 (ANTHROPIC_API_KEY 없음)",
        )
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


# ── 나이대별 어휘/문장 난이도 규칙 ──────────────────────────────
# 나이 숫자만 프롬프트에 넣으면 모델이 일관되게 반응하지 않을 수 있어서,
# 나이대별로 구체적인 행동 규칙을 문장으로 만들어 시스템 프롬프트에 덧붙인다.
def _age_band_guidance(age: Optional[int]) -> str:
    if age is None:
        return ""
    if age <= 6:
        band = (
            "아주 쉬운 낱말만 쓰고, 한 문장에 6~8단어를 넘지 않게 최대한 짧게 써줘. "
            "그림책을 읽어주듯 다정하고 단순하게 설명해줘."
        )
    elif age <= 9:
        band = (
            "초등학교 저학년이 이해할 수 있는 쉬운 낱말을 쓰고, 문장은 짧고 명확하게 써줘. "
            "어려운 한자어나 전문 용어는 피해줘."
        )
    elif age <= 12:
        band = (
            "초등학교 고학년 수준의 어휘를 써도 괜찮지만, 여전히 쉬운 설명을 우선해줘. "
            "필요하면 짧은 예시를 하나 들어줘."
        )
    elif age <= 15:
        band = (
            "중학생 수준의 어휘와 개념어를 사용해도 좋아. "
            "논리적인 이유를 조금 더 자세히 설명해줘."
        )
    else:
        band = (
            "고등학생 이상 수준으로, 필요하면 전문 용어를 써도 괜찮아. "
            "다만 여전히 친근한 말투는 유지해줘."
        )
    return f" 지금 대화하는 아이는 {age}살이야. {band}"


# ── 질문인지 확인하기 ──────────────────────────────────────────
class CheckQuestionRequest(BaseModel):
    question: str
    age: Optional[int] = None


class CheckQuestionResult(BaseModel):
    is_question: bool
    message: str


@router.post("/check-question", response_model=CheckQuestionResult)
async def check_question(body: CheckQuestionRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요")

    client = _get_client()
    try:
        response = await client.messages.parse(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=(
                "너는 초등학생이 쓴 한국어 문장이 '궁금한 것을 스스로 묻는 질문'인지 "
                "확인해주는 다정한 선생님이야. 짧고 서투른 문장이어도 궁금증을 담고 "
                "있다면 질문으로 인정해줘. 질문이 맞으면 그 질문이 무엇을 궁금해하는지 "
                "한두 문장으로 다정하게 짚어주고, 질문이 아니면(인사말, 명령문, 단순 "
                "서술문 등) 왜 질문이 아닌지와 어떻게 고치면 좋을지 초등학생이 이해할 "
                "수 있게 알려줘." + _age_band_guidance(body.age)
            ),
            messages=[{"role": "user", "content": question}],
            output_format=CheckQuestionResult,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")

    return response.parsed_output


# ── 답변 조각을 질문으로 종합하기 ──────────────────────────────
class ComposeQuestionRequest(BaseModel):
    original_question: str
    framework_label: str
    answers: List[str]
    age: Optional[int] = None


class ComposeQuestionResult(BaseModel):
    question: str


@router.post("/compose-question", response_model=ComposeQuestionResult)
async def compose_question(body: ComposeQuestionRequest):
    answers = [a.strip() for a in body.answers if a.strip()]
    if not answers:
        raise HTTPException(status_code=400, detail="답변을 입력해주세요")

    client = _get_client()
    try:
        response = await client.messages.parse(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=(
                f"너는 초등학생이 '{body.framework_label}' 방식으로 채운 답변 조각들을 "
                "자연스럽고 완전한 한 문장짜리 질문으로 다듬어주는 선생님이야. 원래 "
                "질문의 의도를 유지하면서 아이가 직접 쓴 답변들을 최대한 살려서 이어줘. "
                "존댓말이 아니라 아이 말투(반말 의문문)로 쓰고, 결과는 물음표로 끝나는 "
                "한 문장이어야 해." + _age_band_guidance(body.age)
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"원래 질문: {body.original_question}\n"
                    f"답변 조각들: {', '.join(answers)}"
                ),
            }],
            output_format=ComposeQuestionResult,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")

    return response.parsed_output
