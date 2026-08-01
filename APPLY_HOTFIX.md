# v5.1.2 → v5.1.3 덱빌더 핫픽스 적용

1. 이 ZIP의 내용물을 기존 v5.1.2 GitHub 저장소 루트에 덮어씁니다.
2. Git commit 후 push합니다.
3. Railway 자동 재배포가 끝나면 `/api/health`의 version이 `5.1.3-railway`인지 확인합니다.
4. 대기실에 `덱빌더` 버튼이 표시되는지 확인합니다.

변경 파일:

- `app/server.py`
- `app/static/app.js`
- `app/templates/index.html`
- `VERSION.txt`

환경 변수와 `/data` Volume 설정은 변경할 필요가 없습니다.
