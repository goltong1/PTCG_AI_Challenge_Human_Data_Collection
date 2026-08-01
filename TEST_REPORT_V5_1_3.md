# CABT Railway v5.1.3 테스트 보고서

검증 항목:

- Python 전체 문법 검사
- JavaScript 문법 검사
- 공개 모드 `/api/config`에서 덱 저장 범위가 `browser`로 표시되는지 확인
- 공개 페이지에서 덱빌더 버튼이 숨김 클래스 없이 표시되는지 확인
- 프리셋 60장 덱 불러오기
- `/api/decks/inspect`의 카드 검증 및 상세 정보 반환
- 브라우저 저장 덱 카드 배열로 AI 대전 시작 및 종료
- 서로 다른 두 세션이 브라우저 저장 덱 카드 배열로 빠른 매칭
- 첫 번째 대기자는 워커 0개, 실제 PvP 매치 생성 시 워커 1개 사용
- PvP 종료 후 워커 0개 반환
- localStorage 저장·복원·사용 payload 생성 로직 단위 검사
- 최종 ZIP 무결성 검사

참고: 작업 환경의 Chromium 정책이 localhost 페이지 접근을 차단하여 실제 브라우저 클릭 자동화는 수행하지 못했습니다. 대신 서버 통합 테스트, JavaScript 구문 검사, localStorage 핵심 로직 단위 검사를 수행했습니다.
