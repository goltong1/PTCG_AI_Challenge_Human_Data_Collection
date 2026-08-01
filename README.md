# CABT Online Arena · Railway Edition v5.1.1

Railway에 Dockerfile로 바로 배포하는 전용 소스 패키지입니다.

기능:

- 사람 대 AI 대전
- 플레이어 빠른 매칭 및 6자리 친구 방
- 플레이어별 비공개 패 정보
- 프라이즈, 부착 에너지, 트래시 이미지 중심 UI
- 공식 JSON 및 기록 ZIP 생성
- `/admin` 결과 관리 화면
- `/data` Railway Volume에 결과 영구 저장

## 가장 빠른 배포

1. 이 폴더의 **내용물 전체**를 GitHub 저장소 루트에 업로드합니다.
2. Railway에서 `New Project → Deploy from GitHub repo`를 선택합니다.
3. 서비스 Variables에 `CABT_ADMIN_TOKEN`을 등록합니다.
4. Volume을 만들고 Mount Path를 `/data`로 설정합니다.
5. `Settings → Networking → Generate Domain`을 누릅니다.
6. `https://생성주소/api/health`에서 `ok: true`를 확인합니다.

전체 절차와 오류 해결은 [`DEPLOY_RAILWAY.md`](DEPLOY_RAILWAY.md)를 참고하세요.

## 무료 플랜 기본 용량

무료 플랜의 0.5GB RAM에 맞춰 **실제 게임 워커**를 1개로 제한했습니다. PvP 게임 1개에는 플레이어 2명이 함께 참가할 수 있으며, 매칭 큐와 친구 방에서 기다리는 사용자는 게임 워커 슬롯을 사용하지 않습니다. AI 게임 1개 또는 PvP 경기 1개가 실제로 진행될 때만 슬롯을 사용합니다.

Hobby 이상에서 동시 게임 수를 늘리려면 `CABT_MAX_ACTIVE_WORKERS`, `CABT_MAX_SESSIONS`, `CABT_MAX_PVP_MATCHES`를 함께 조정하세요.


## v5.1.1 매칭 슬롯 패치

- 사용자가 빠른 매칭 또는 친구 방에 들어갈 때 같은 브라우저 세션에 남아 있던 AI/리플레이 워커를 자동 종료합니다.
- 첫 번째 플레이어가 큐에 들어간 상태는 `active worker 0`으로 유지됩니다.
- 두 번째 플레이어가 참가해 실제 경기가 생성되는 순간에만 `active worker 1`이 됩니다.
- 같은 브라우저의 일반 탭끼리는 세션 쿠키를 공유하므로 테스트할 때는 서로 다른 브라우저, 시크릿 창 또는 서로 다른 기기를 사용하세요.
