# v5.1.1 → v5.1.2 적용

이 ZIP의 내용을 기존 GitHub 저장소 최상단에 덮어씁니다.

```bash
git add .
git commit -m "Fix disconnected sessions holding Railway worker slot"
git push
```

Railway가 자동 재배포합니다. 기존 `/data` Volume과 관리자 토큰 설정은 유지됩니다.

권장 Variables:

```text
CABT_CLIENT_DISCONNECT_SECONDS=45
CABT_CLIENT_DISCONNECT_GRACE_SECONDS=15
```
