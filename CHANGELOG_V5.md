# CABT Online Battle Arena v5

## Player vs Player

- 무작위 빠른 매칭
- 6자리 코드 기반 친구 방
- 플레이어별 60장 덱과 닉네임
- 서버 권한형 턴·선택지 검증
- 상대 패를 전송하지 않는 좌석별 상태 응답
- 게임별 CABT 네이티브 워커 프로세스 격리
- 동일 브라우저 세션 재접속
- 상대 연결·퇴장 표시
- PvP 기록 ZIP, 공식 JSON, 공식 뷰어 다운로드
- 활성 게임 수와 매칭 대기자 수 제한

## Deployment

- Fly.io 도쿄 리전용 `fly.toml`
- 일반 VPS용 Docker Compose + Caddy 구성
- `/data` 영구 저장소
- 단일 인스턴스 운영 지침

## Compatibility

- 기존 Human vs AI 대전
- AI 결과 제출 및 관리자 페이지
- 리플레이 뷰어
- Windows 휴대용 실행
