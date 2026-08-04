# v5.1.4 · 10명 동시접속 패치

"
        "기본 제한을 다음과 같이 변경했습니다.

"
        "- `CABT_MAX_ACTIVE_WORKERS=10`: AI 경기 기준 최대 10개 활성 워커
"
        "- `CABT_MAX_SESSIONS=10`: AI/리플레이 브라우저 세션 최대 10개
"
        "- `CABT_MAX_PVP_MATCHES=5`: PvP 최대 5경기, 즉 최대 10명
"
        "- `CABT_MAX_PVP_WAITERS=20`: 매칭 대기 및 친구 방 여유 인원 20명

"
        "## Railway에서 반드시 확인할 것

"
        "Railway Variables에 기존 값 `1`이 등록되어 있으면 Dockerfile보다 우선합니다. "
        "아래 네 값을 직접 변경하거나 기존 변수를 삭제한 뒤 재배포하세요.

"
        "```env
"
        "CABT_MAX_ACTIVE_WORKERS=10
"
        "CABT_MAX_SESSIONS=10
"
        "CABT_MAX_PVP_MATCHES=5
"
        "CABT_MAX_PVP_WAITERS=20
"
        "```

"
        "## 자원 주의

"
        "각 활성 게임은 별도 Python 프로세스와 엔진 복사본을 사용합니다. "
        "Railway 0.5 GB급 인스턴스에서는 10개의 AI 게임을 동시에 실행할 때 메모리 부족으로 재시작될 수 있습니다. "
        "그 경우 `CABT_MAX_ACTIVE_WORKERS=5`로 낮추되, PvP 5경기/10명만 목표라면 워커 5개로 충분합니다.
"
        