# v5.1.1 회귀 테스트 결과

테스트 환경은 Railway 무료 플랜과 같은 제한값을 사용했습니다.

```text
CABT_MAX_ACTIVE_WORKERS=1
CABT_MAX_SESSIONS=1
CABT_MAX_PVP_MATCHES=1
CABT_MAX_PVP_WAITERS=12
```

## 기존 v5.1.0 재현

1. 플레이어 1이 AI 게임 시작: `active worker = 1`
2. 플레이어 1이 빠른 매칭 큐 참가: `active worker = 1`로 남음
3. 플레이어 2가 빠른 매칭 참가: HTTP 503

```text
현재 서버 게임 처리 한도(1게임)에 도달했습니다.
```

## 수정판 v5.1.1 빠른 매칭

1. 플레이어 1이 AI 게임 시작: `active worker = 1`
2. 플레이어 1이 빠른 매칭 큐 참가: `active worker = 0`
3. 플레이어 2가 빠른 매칭 참가: HTTP 200, `status = matched`
4. 경기 생성 후: `active worker = 1`, `active_matches = 1`
5. 두 플레이어의 `match_id`가 동일하고 좌석은 서로 다름

## 수정판 v5.1.1 친구 방

1. 방장이 AI 게임 시작: `active worker = 1`
2. 방장이 친구 방 생성: `active worker = 0`
3. 게스트가 방 코드로 참가: HTTP 200, `status = matched`
4. 경기 생성 후: `active worker = 1`, `active_matches = 1`
5. 양쪽에서 자신의 좌석 기준으로 비공개 상태 조회 성공

## 정적 검사

- Python 구문 검사 통과
- JavaScript 구문 검사 통과
- ZIP 무결성 검사 통과 예정
