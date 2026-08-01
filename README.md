# CABT Online Arena · Railway Edition v5.1

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

무료 플랜의 0.5GB RAM에 맞춰 전체 게임 워커를 1개로 제한했습니다. 즉, 한 시점에 AI 게임 1개 또는 PvP 게임 1개가 실행됩니다. 진행 중 게임이 끝나면 다음 게임을 시작할 수 있습니다.

Hobby 이상에서 동시 게임 수를 늘리려면 `CABT_MAX_ACTIVE_WORKERS`, `CABT_MAX_SESSIONS`, `CABT_MAX_PVP_MATCHES`를 함께 조정하세요.
