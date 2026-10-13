"""
AI 라우터 (질문 분석·구체화 퀴즈 생성 / 문장 쓰기 첨삭 / 심화 질문 추천)
─────────────────────────────────────────────────────────────
Claude API를 이용해 세 가지를 처리합니다.
  1. POST /ai/analyze-question          : 아이가 쓴 문장이 질문인지 확인하고, 질문이면
                                           그 질문에서 막연한 부분 세 곳을 AI가 직접 골라
                                           구체화 퀴즈 3단계 + 실제 설명을 한 번에 생성
  2. POST /ai/check-composition         : 아이가 답 조각들을 직접 이어 쓴 문장(1~2개)의
                                           지칭 일관성/문법/맞춤법을 첨삭
  3. POST /ai/suggest-advanced-question : 원래 질문보다 한 단계 더 깊은 심화 질문 추천

모든 호출은 _ask_structured() 한 곳을 거칩니다 (모델/effort/프롬프트 캐싱/실패 처리).
ANTHROPIC_API_KEY 는 app/config.py 의 settings 에서만 읽습니다.
"""
from typing import List, Optional, Type, TypeVar

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from ..config import settings

router = APIRouter()

MODEL = "claude-haiku-5-5"

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
        # 나이는 선택 정보라(비로그인·미입력) 없으면 초등학생 기본 수준으로 맞춘다.
        return (
            "지금 대화하는 아이는 초등학생이야. 초등학생이 이해할 수 있는 쉬운 "
            "낱말을 쓰고, 문장은 짧고 명확하게 써줘."
        )
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
    return f"지금 대화하는 아이는 {age}살이야. {band}"


ResultT = TypeVar("ResultT", bound=BaseModel)


async def _ask_structured(
    *,
    system: str,
    age: Optional[int],
    user_content: str,
    output_format: Type[ResultT],
    max_tokens: int,
) -> ResultT:
    # 고정 시스템 프롬프트에만 캐시 표시를 하고, 나이별 문장은 그 뒤 별도 블록에
    # 둔다. 나이 문장이 고정 프롬프트 안에 섞여 있으면 나이마다 캐시가 갈라진다.
    # (Haiku 5.5의 최소 캐시 길이는 512토큰 — 그보다 짧은 프롬프트는 그냥 캐시 안 됨)
    system_blocks = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    guidance = _age_band_guidance(age)
    if guidance:
        system_blocks.append({"type": "text", "text": guidance})

    client = _get_client()
    try:
        response = await client.messages.parse(
            model=MODEL,
            # Haiku 5.5는 기본으로 생각(thinking)을 하고 그 토큰도 max_tokens에
            # 포함된다. effort low로 생각을 최소화하고, 여유를 둔 상한을 준다.
            max_tokens=max_tokens,
            output_config={"effort": "low"},
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
            output_format=output_format,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")
    except ValidationError:
        raise HTTPException(status_code=502, detail="AI 응답 형식이 올바르지 않아요")

    # Haiku 5.5는 서버 측 fallback 없이 요청을 거절할 수 있고, max_tokens에
    # 걸리면 JSON이 끊긴다. 둘 다 parsed_output이 None이 된다.
    if response.stop_reason == "refusal":
        raise HTTPException(status_code=502, detail="AI가 이 요청에 답하지 않았어요")
    parsed = response.parsed_output
    if parsed is None:
        raise HTTPException(status_code=502, detail="AI 응답을 끝까지 받지 못했어요")
    return parsed


# ── 질문 분석 + 구체화 퀴즈 생성 (한 번의 호출) ─────────────────
# 예전에는 아이가 '육하원칙'/'원인과 결과' 중 하나를 고르게 했지만, 배우려는
# 건 틀의 이름이 아니라 "막연한 질문을 구체적으로 만드는 법"이다. 그래서 질문마다
# 무엇이 빠져 있거나 막연한지를 AI가 직접 판단해 세 곳을 고른다.
# 질문 확인과 퀴즈 생성을 한 번에 하는 이유: 사이에 아이가 고를 것이 없어졌고,
# 비용의 대부분은 출력(퀴즈) 토큰이라 호출을 나누면 입력·요청만 중복된다.
class QuizOption(BaseModel):
    id: str
    label: str
    value: str


class QuizStep(BaseModel):
    id: str
    title: str
    desc: str
    options: List[QuizOption]
    correct_id: str
    wrong_msg: str
    correct_msg: str


class AnalyzeQuestionRequest(BaseModel):
    question: str
    age: Optional[int] = None


class AnalyzeQuestionResult(BaseModel):
    is_question: bool
    message: str
    quizzes: List[QuizStep]
    answer: str


_ANALYZE_SYSTEM = (
    "너는 초등학생이 짧고 막연하게 던진 질문을, 아이 스스로 더 구체적인 "
    "질문으로 다듬도록 돕는 다정한 선생님이야.\n\n"
    "먼저 질문인지 확인해. 짧고 서투른 문장이어도 궁금증을 담고 있으면 "
    "질문으로 인정해줘. 질문이 아니면(인사말, 명령문, 단순 서술문 등) "
    "is_question을 false로 하고, message에 왜 질문이 아닌지와 어떻게 "
    "고치면 좋을지 알려줘. 이때 quizzes는 빈 배열, answer는 빈 문자열로 둬.\n\n"
    "질문이 맞으면 is_question을 true로 하고, message에 이 질문이 무엇을 "
    "궁금해하는지 한두 문장으로 짚어줘. 그다음 이 질문을 더 구체적으로 "
    "만들려면 무엇이 빠져 있거나 막연한지 직접 판단해서, 가장 중요한 "
    "세 가지를 골라 퀴즈 3단계로 만들어줘. 정해진 틀은 없어. 예를 들어 "
    "무엇에 대한 질문인지(대상), 언제·어디서의 일인지, 막연한 낱말을 더 "
    "정확한 낱말로 바꾸기, 어떤 원인이나 과정이 궁금한지, 어떤 조건이나 "
    "범위에서의 일인지 같은 것 중에서 이 질문에 정말 필요한 것만 골라. "
    "질문에 이미 분명히 들어 있는 내용은 다시 묻지 마.\n\n"
    "각 단계의 title은 무엇을 구체화하는지 드러나게 번호를 붙여 써줘(예: "
    "'1. 어디에서 일어나는 일인지 정하기'). id는 영문 소문자로 써줘. 각 "
    "단계마다 객관식 보기를 3개(정답 1개 + 그럴듯한 오답 2개) 만들고, "
    "보기에는 id(영문 소문자/숫자), label(화면에 보여줄 설명), value(답 "
    "조각으로 쓰일 짧은 표현)를 채워줘. correct_id는 정답 보기의 id와 같아야 "
    "해. wrong_msg는 틀렸을 때 다시 생각해보게 다정하게 격려하는 문장, "
    "correct_msg는 맞았을 때 짧게 칭찬하는 문장으로 써줘. 마지막으로 세 "
    "조각을 넣어 구체적으로 다듬어진 질문에 대한 실제 설명을 answer에 "
    "2~3문장으로 써줘. 모든 보기와 설명은 실제 사실에 기반해야 하고, "
    "존댓말이 아니라 아이에게 말하듯 친근한 말투로, 쉬운 표현으로 써줘."
)


@router.post("/analyze-question", response_model=AnalyzeQuestionResult)
async def analyze_question(body: AnalyzeQuestionRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요")

    return await _ask_structured(
        system=_ANALYZE_SYSTEM,
        age=body.age,
        user_content=f"아이의 질문: {question}",
        output_format=AnalyzeQuestionResult,
        max_tokens=4096,
    )


# ── 아이가 직접 쓴 문장 첨삭하기 ────────────────────────────────
# 답 조각을 하나의 완벽한 문장으로 합치는 건 쉬운데, 아이들은 조각 2개를
# 자기 힘으로 이어서 자연스러운 글로 쓰는 걸 유독 어려워한다(지칭이 꼬이거나
# 앞뒤 문장이 안 맞는 경우가 많음). 그래서 꼭 한 문장일 필요는 없고, 두 문장을
# 써도 되니 지칭 일관성 / 문법 / 맞춤법만 봐준다.
class CheckCompositionRequest(BaseModel):
    pieces: List[str]
    attempt: str
    age: Optional[int] = None


class CheckCompositionResult(BaseModel):
    is_correct: bool
    message: str


_COMPOSITION_SYSTEM = (
    "너는 초등학생의 글쓰기를 봐주는 다정한 선생님이야. 아이는 여러 개의 "
    "짧은 생각 조각들을 받아서, 그 내용을 살려 자기 힘으로 글을 이어 "
    "썼어. 한 문장으로 합치지 않고 두 문장으로 나눠 써도 전혀 문제없어 "
    "— 억지로 한 문장으로 만들라고 하지 마. 대신 다음만 확인해줘:\n"
    "1. 지칭 일관성: 앞 문장에서 말한 대상을 뒷 문장에서 다시 가리킬 때 "
    "헷갈리지 않게 잘 이어지는지 (예: 주어가 갑자기 바뀌거나, '그것/이것' "
    "이 무엇을 가리키는지 불분명한 경우)\n"
    "2. 문법: 문장 구조나 조사 사용이 자연스러운지\n"
    "3. 맞춤법: 띄어쓰기나 철자가 틀린 곳이 있는지\n\n"
    "문제가 있으면 is_correct를 false로 하고, 어느 부분이 왜 어색한지 "
    "구체적으로 짚어서 초등학생이 이해할 수 있게 다정하게 설명해줘 "
    "(예: '앞에서는 공룡이라고 했는데 뒤에서 갑자기 다른 걸 가리키는 "
    "것처럼 보여' 처럼). 문제가 없으면 is_correct를 true로 하고 짧게 "
    "칭찬해줘."
)


@router.post("/check-composition", response_model=CheckCompositionResult)
async def check_composition(body: CheckCompositionRequest):
    attempt = body.attempt.strip()
    if not attempt:
        raise HTTPException(status_code=400, detail="작성한 문장을 입력해주세요")
    pieces = [p.strip() for p in body.pieces if p.strip()]

    return await _ask_structured(
        system=_COMPOSITION_SYSTEM,
        age=body.age,
        user_content=(
            f"주어진 생각 조각들: {', '.join(pieces) if pieces else '(없음)'}\n"
            f"아이가 쓴 글: {attempt}"
        ),
        output_format=CheckCompositionResult,
        max_tokens=2048,
    )


# ── 심화 질문 추천하기 ──────────────────────────────────────────
class SuggestAdvancedQuestionRequest(BaseModel):
    question: str
    age: Optional[int] = None


class SuggestAdvancedQuestionResult(BaseModel):
    question: str


_ADVANCED_SYSTEM = (
    "너는 초등학생이 궁금해하는 것을 한 단계 더 깊게 넓혀주는 선생님이야. "
    "아이가 이미 물어본 질문을 보고, 그 질문에 답을 알고 난 다음에 자연스럽게 "
    "이어서 궁금해질 만한 심화 질문을 하나만 만들어줘. 같은 주제를 다른 각도로 "
    "파고들거나, 한 단계 더 깊은 '왜'나 '어떻게'를 묻는 질문이면 좋아. 존댓말이 "
    "아니라 아이 말투(반말 의문문)로, 물음표로 끝나는 한 문장으로 만들어줘."
)


@router.post(
    "/suggest-advanced-question", response_model=SuggestAdvancedQuestionResult
)
async def suggest_advanced_question(body: SuggestAdvancedQuestionRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요")

    return await _ask_structured(
        system=_ADVANCED_SYSTEM,
        age=body.age,
        user_content=question,
        output_format=SuggestAdvancedQuestionResult,
        max_tokens=1024,
    )
