# Vertex AI Claude to OpenAI SDK Proxy

이 저장소는 기존 애플리케이션들에서 널리 쓰이는 표준 **OpenAI SDK**를 전혀 수정하지 않고, Google Cloud Vertex AI 환경에 배포된 **Anthropic Claude 모델(예: Claude Opus 4.6)** 을 자유롭게 호출할 수 있도록 도와주는 중계 서버(라우터) 샘플 코드입니다.

## 목적 및 배경

Google Cloud의 자체 호환 엔드포인트(`.googleapis.com/.../openapi`)는 내부의 Gemini 모델들 및 특정 오픈 모델들만 네이티브하게 지원합니다. 따라서 Vertex AI 상의 Claude 파트너 모델들은 OpenAI SDK로 직접 다이렉트 통신(`FAILED_PRECONDITION: Chat completions not supported.`)이 불가합니다.

이를 완벽에 가깝게 해결하는 방법으로, **LLM 요청/응답 규격을 실시간으로 변환하는 Middleware Gateway Server(`proxy.py`)를 중앙에 두는 형태**가 엔터프라이즈 환경 등에서 표준 아키텍처로써 사용되고 있습니다.

## 데이터 흐름 (Workflow)

1. **Client**(`client.py`)는 기본 `openai.OpenAI()`를 통해 아무런 스키마 변형 없이 표준 `/chat/completions` 요청을 발송합니다.
2. **Proxy Router Server**(`proxy.py`)가 이 요청을 낚아채어, 다음 과정을 거칩니다:
   - System 프롬프트 분리
   - Tool/Function JSON Schema 및 Role 구조를 Anthropic Messages API 형식으로 형변환(Translation)
   - Application Default Credentials에서 갱신한 GCP Bearer 토큰으로 헤더를 변경하여 Vertex 측의 `:rawPredict` Endpoint로 전달
3. **Vertex AI** 에서는 변환된 명령을 Claude에게 넘겨 결과를 JSON 형식으로 리턴합니다.
4. **Proxy Router Server**는 받은 결과물을 다시 역변환하여, 기존 클라이언트가 기대하는 `chatcmpl-...` ID 규격 및 `choices[]` 응답 스키마로 가공하여 반환합니다.

## 퀵 스타트 가이드 (테스트 방법)

### 1단계: 프로젝트 인증 및 세팅

사전에 사용중인 장비(로컬 컴퓨터 등)에 Google Cloud 자격증명이 되어 있어야 합니다.

```bash
gcloud auth application-default login
```

> [!IMPORTANT]
> 실제로 서비스를 돌리시기 전에 `proxy.py` 파일 내에 위치해 있는 `VERTEX_PROJECT = "YOUR_PROJECT_ID"` 값을 본인의 실제 Google Cloud Project ID로 꼭 변경해주세요.

### 2단계: 가상 환경 설정 및 패키지 설치

Python 의존성 관리 도구인 `uv`나 `pip`를 사용해 패키지를 설치합니다.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3단계: 프록시(라우터) 서버 실행

창을 하나 열고 아래 명령어로 FastAPI 기반 릴레이 서버를 기동합니다. (기본 Port: 8000)

```bash
uv run uvicorn proxy:app --host 0.0.0.0 --port 8000
```

### 4단계: OpenAI 클라이언트로 테스트 결과 확인

새로운 터미널 창을 열어서 클라이언트 스크립트를 마저 실행해보세요

```bash
uv run python client.py
```

클라이언트는 8000번 포트의 Proxy를 바라보고 통신하지만, 로그 상으로는 실제 클라우드를 돌고 온 Claude의 대답이 떨어집니다.
