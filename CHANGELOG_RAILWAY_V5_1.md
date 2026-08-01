# v5.1 Railway 변경사항

- Railway 전용 Docker/설정/배포 문서로 단순화
- Railway Volume 경로 자동 감지
- AI 세션과 PvP가 공유하는 전역 워커 한도 추가
- 무료 플랜 기본 전역 워커 1개로 메모리 보호
- AI 게임/리플레이 종료 시 워커 즉시 반환
- 워커 부족 시 빠른 매칭 대기열과 친구 방 복원
- `/api/health`에 워커 사용량과 Volume 정보 추가
- `tini`를 사용해 자식 프로세스 종료 및 좀비 프로세스 정리 강화
- Linux 배포에 불필요한 Windows/macOS/ARM 네이티브 바이너리 제거
