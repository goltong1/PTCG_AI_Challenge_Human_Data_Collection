# CABT Web Arena · Railway 배포

이 버전은 브라우저에서 여러 사용자가 각각 독립된 CABT 엔진 프로세스로 AI와 대전하고, 완료된 결과를 서버의 영구 저장소로 전송하도록 구성되어 있습니다.

## 권장 구성

- 애플리케이션: Railway 단일 서비스, Dockerfile 배포
- 저장소: 서비스에 연결한 Volume을 `/data`에 마운트
- 인스턴스 수: 1개
- 결과 보관 위치: `/data/submissions`
- 원본 게임 기록: `/data/records`
- 관리자 페이지: `https://배포주소/admin`

CABT 네이티브 엔진과 사용자 세션은 한 인스턴스의 프로세스에 연결됩니다. 볼륨이 인스턴스별로 마운트되므로 처음에는 **replica 1개**로 운영하세요.

## 1. GitHub 저장소 준비

이 폴더의 내용을 새 GitHub 저장소 루트에 올립니다. 다음 파일이 루트에 있어야 합니다.

```text
Dockerfile
railway.json
requirements.txt
app/
```

기존 `app/records/`는 Docker 이미지에 포함되지 않습니다.

## 2. Railway 프로젝트 생성

1. Railway에서 **New Project → Deploy from GitHub repo**를 선택합니다.
2. 이 저장소를 선택합니다.
3. Railway가 루트의 `Dockerfile`을 인식하도록 둡니다.
4. 배포 전에 아래 환경 변수를 등록합니다.

| 변수 | 권장값 | 설명 |
|---|---:|---|
| `CABT_PUBLIC_MODE` | `1` | 사용자별 독립 게임 프로세스 사용 |
| `CABT_ENABLE_RESULT_SUBMISSION` | `1` | 완료 기록 전송 버튼 활성화 |
| `CABT_DATA_DIR` | `/data` | 영구 볼륨 저장 경로 |
| `CABT_ADMIN_TOKEN` | 긴 임의 문자열 | `/admin`에서 결과를 조회할 비밀 토큰 |
| `CABT_MAX_SESSIONS` | `6` | 동시에 유지할 게임 프로세스 수 |
| `CABT_SESSION_IDLE_SECONDS` | `3600` | 미사용 세션 종료 시간 |
| `CABT_COOKIE_SECURE` | `1` | HTTPS 세션 쿠키 강제 |

`CABT_ADMIN_TOKEN`은 최소 32자 이상의 무작위 문자열을 권장합니다.

## 3. 영구 Volume 연결

1. Railway 서비스의 **Volumes**에서 새 Volume을 추가합니다.
2. Mount Path를 `/data`로 설정합니다.
3. 서비스를 다시 배포합니다.

볼륨을 연결하지 않으면 재배포 또는 재시작 때 수집 결과가 사라질 수 있습니다.

## 4. 공개 주소 생성

서비스의 **Networking → Generate Domain**을 선택합니다. 생성된 HTTPS 주소의 `/api/health`에 접속해 다음과 비슷한 응답이 보이면 정상입니다.

```json
{"ok":true,"public_mode":true}
```

## 5. 결과 확인

1. `https://배포주소/admin`을 엽니다.
2. Railway에 설정한 `CABT_ADMIN_TOKEN`을 입력합니다.
3. 전송된 경기의 승패, AI, 덱, 액션 수를 확인합니다.
4. `ZIP` 버튼으로 `official_replay.json`, `replay_visualize.json`, `transitions.jsonl`, 덱 파일이 포함된 전체 기록을 내려받습니다.

## 공개판 보안 설정

공개 모드에서는 다음 기능이 자동으로 차단됩니다.

- 사용자의 임의 AI ZIP 업로드 및 서버 실행
- 서버 공용 덱 저장·삭제
- 카드 이미지 서버 재검색

번들에 포함할 AI는 배포 전에 `app/agents/`에 직접 넣으세요. 제출 AI 코드는 서버 권한으로 실행되므로 신뢰할 수 있는 코드만 포함해야 합니다.

## 로컬 Docker 검증

```bash
docker compose up --build
```

- 아레나: `http://localhost:8765`
- 관리자: `http://localhost:8765/admin`
- 예제 관리자 토큰: `change-this-admin-token`

실제 공개 배포 전에는 예제 토큰을 반드시 변경하세요.

## 서버 용량 조정

각 활성 게임은 독립 Python 프로세스와 CABT 네이티브 엔진을 사용합니다. 메모리가 부족하거나 응답이 느리면 `CABT_MAX_SESSIONS`를 먼저 낮추고, Railway 인스턴스 메모리를 올리세요. 한 사용자가 페이지를 닫아도 세션은 유휴 제한 시간까지 유지되며 이후 자동 종료됩니다.
