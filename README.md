# CABT GitHub Pages Static

정적 GitHub Pages 배포용 패키지입니다.

## 제공 기능

- Windows 로컬 AI 대전판 다운로드 링크
- CABT 기록 ZIP 및 JSON 브라우저 리플레이 뷰어
- 부착 에너지·프라이즈·트래시 중심의 개선 UI
- 최근 리플레이를 브라우저 IndexedDB에 로컬 저장
- 서버 API, 온라인 매칭, 결과 업로드 기능 없음

## 중요한 제한

GitHub Pages는 HTML/CSS/JavaScript 같은 정적 파일만 실행합니다. 현재 CABT 게임 규칙 엔진은 Windows DLL/Linux SO와 Python AI에 의존하므로 GitHub Pages 브라우저 안에서 실시간 AI 대전을 실행할 수 없습니다. 실시간 대전은 별도 제공되는 `CABT_Local_AI_Arena_v6.zip`을 Windows에서 실행합니다.

## 배포

`DEPLOY_GITHUB_PAGES.md`를 참고하세요.
