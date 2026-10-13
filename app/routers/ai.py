"""
AI 라우터 (질문 검증 / 육하원칙 종합 / 문장 쓰기 첨삭 / 퀴즈 생성)
─────────────────────────────────────────────────────────────
Claude API를 이용해 다섯 가지를 처리합니다.
  1. POST /ai/check-question            : 아이가 쓴 문장이 진짜 질문인지 확인
  2. POST /ai/compose-question          : 프레임워크(예: 육하원칙) 답변 조각들을
                                           자연스러운 한 문장 질문으로 종합
  3. POST /ai/check-composition         : 아이가 답 조각들을 직접 이어 쓴 문장(1~2개)의
                                           지칭 일관성/문법/맞춤법을 첨삭
  4. POST /ai/suggest-advanced-question : 원래 질문보다 한 단계 더 깊은 심화 질문 추천
  5. POST /ai/generate-quiz             : 아이의 실제 질문 주제에 맞는 3단계 퀴즈 생성
                                           (육하원칙 / 원인과 결과)

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


@router.post("/check-composition", response_model=CheckCompositionResult)
async def check_composition(body: CheckCompositionRequest):
    attempt = body.attempt.strip()
    if not attempt:
        raise HTTPException(status_code=400, detail="작성한 문장을 입력해주세요")
    pieces = [p.strip() for p in body.pieces if p.strip()]

    client = _get_client()
    try:
        response = await client.messages.parse(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=(
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
                "칭찬해줘." + _age_band_guidance(body.age)
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"주어진 생각 조각들: {', '.join(pieces) if pieces else '(없음)'}\n"
                    f"아이가 쓴 글: {attempt}"
                ),
            }],
            output_format=CheckCompositionResult,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")

    return response.parsed_output


# ── 심화 질문 추천하기 ──────────────────────────────────────────
class SuggestAdvancedQuestionRequest(BaseModel):
    question: str
    age: Optional[int] = None


class SuggestAdvancedQuestionResult(BaseModel):
    question: str


@router.post(
    "/suggest-advanced-question", response_model=SuggestAdvancedQuestionResult
)
async def suggest_advanced_question(body: SuggestAdvancedQuestionRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요")

    client = _get_client()
    try:
        response = await client.messages.parse(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=(
                "너는 초등학생이 궁금해하는 것을 한 단계 더 깊게 넓혀주는 선생님이야. "
                "아이가 이미 물어본 질문을 보고, 그 질문에 답을 알고 난 다음에 자연스럽게 "
                "이어서 궁금해질 만한 심화 질문을 하나만 만들어줘. 같은 주제를 다른 각도로 "
                "파고들거나, 한 단계 더 깊은 '왜'나 '어떻게'를 묻는 질문이면 좋아. 존댓말이 "
                "아니라 아이 말투(반말 의문문)로, 물음표로 끝나는 한 문장으로 만들어줘."
                + _age_band_guidance(body.age)
            ),
            messages=[{"role": "user", "content": question}],
            output_format=SuggestAdvancedQuestionResult,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI 호출 실패: {e}")

    return response.parsed_output


# ── 질문 주제에 맞는 퀴즈 생성하기 ──────────────────────────────
# 예전에는 "공룡은 왜 사라졌어?" 하나의 예시로 퀴즈 내용이 고정돼 있었다.
# 아이가 실제로 어떤 질문을 입력하든 그 주제에 맞는 3단계 퀴즈가 나오도록
# 매번 AI가 새로 만든다.
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


class GenerateQuizRequest(BaseModel):
    question: str
    framework: str
    age: Optional[int] = None


class GenerateQuizResult(BaseModel):
    quizzes: List[QuizStep]
    answer: str


_FRAMEWORK_QUIZ_GUIDANCE = {
    "oheukwonchik": (
        "너는 초등학생이 짧은 질문을 '육하원칙' 방식으로 더 또렷하게 다듬도록 "
        "돕는 퀴즈를 만드는 선생님이야. 아이의 원래 질문을 보고 아래 3단계로 "
        "퀴즈를 만들어줘.\n"
        "1단계(id='who'): 질문이 무엇에 대한 것인지(핵심 대상)를 맞히는 문제\n"
        "2단계(id='when'): 질문에서 '언제'에 해당하는 시점/배경을 채우는 문제\n"
        "3단계(id='precise'): 질문에 쓰인 표현을 더 정확하고 사실적인 표현으로 "
        "바꾸는 문제\n"
        "마지막으로 이 세 조각(핵심 대상/시점/정확한 표현)을 '{시점}, {핵심 "
        "대상}은 왜 {정확한 표현}하게 되었을까?' 같은 식으로 합쳤을 때 나올 수 "
        "있는 실제 설명을 answer 필드에 2~3문장으로 써줘."
    ),
    "causeEffect": (
        "너는 초등학생이 짧은 질문을 '원인과 결과' 방식으로 더 또렷하게 다듬도록 "
        "돕는 퀴즈를 만드는 선생님이야. 아이의 원래 질문을 보고 아래 3단계로 "
        "퀴즈를 만들어줘.\n"
        "1단계(id='effect'): 질문에서 이미 벌어진 일(결과)이 무엇인지 확인하는 "
        "문제\n"
        "2단계(id='cause'): 그 결과를 일으켰을 법한 원인 후보를 좁히는 문제\n"
        "3단계(id='mechanism'): 그 원인이 어떻게 결과로 이어지는지 과정/연결고리를 "
        "찾는 문제\n"
        "마지막으로 원인이 어떤 과정을 거쳐 결과로 이어졌는지 실제 설명을 "
        "answer 필드에 2~3문장으로 써줘."
    ),
}


@router.post("/generate-quiz", response_model=GenerateQuizResult)
async def generate_quiz(body: GenerateQuizRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요")
    guidance = _FRAMEWORK_QUIZ_GUIDANCE.get(body.framework)
    if guidance is None:
        raise HTTPException(status_code=400, detail="알 수 없는 프레임워크예요")

    client = _get_client()
    try:
        response = await client.messages.parse(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=(
                guidance + "\n\n"
                "각 단계마다 객관식 보기를 3개 만들어줘(정답 1개 + 그럴듯한 오답 "
                "2개). 보기에는 id(영문 소문자/숫자), label(화면에 보여줄 설명), "
                "value(답 조각으로 쓰일 짧은 표현)를 채워줘. correct_id는 정답 "
                "보기의 id와 같아야 해. wrong_msg는 틀렸을 때 다시 생각해보게 "
                "다정하게 격려하는 문장, correct_msg는 맞았을 때 짧게 칭찬하는 "
                "문장으로 써줘. 모든 보기와 설명은 실제 사실에 기반해야 하고, "
                "존댓말이 아니라 아이에게 말하듯 친근한 말투로, 초등학생이 이해할 "
                "수 있는 쉬운 표현으로 써줘." + _age_band_guidance(body.age)
            ),
            messages=[{"role": "user", "content": f"아이의 질문: {question}"}],
            output_format=GenerateQuizResult,
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
