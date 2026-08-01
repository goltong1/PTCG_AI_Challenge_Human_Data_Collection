# CABT Online Arena v5.1 · Railway 배포 가이드

## 0. 먼저 알아둘 점

- 이 패키지는 Linux Docker 배포 전용입니다.
- 온라인 매칭 상태는 단일 서버 메모리에 있으므로 **Replica는 반드시 1개**로 유지합니다.
- 완료된 기록은 `/data` Volume에 남지만, 배포·재시작 중 진행 중인 게임은 종료됩니다.
- 무료 플랜 기본값은 메모리 보호를 위해 전체 활성 게임 워커 1개입니다.

## 1. GitHub 저장소 만들기

1. GitHub에서 새 저장소를 만듭니다.
2. ZIP을 풀고, 바깥 폴더가 아니라 그 안의 파일을 저장소 루트에 올립니다.
3. 저장소 루트에 아래가 보여야 합니다.

```text
Dockerfile
railway.json
requirements.txt
app/
DEPLOY_RAILWAY.md
```

GitHub 웹 업로드를 써도 되고 Git을 써도 됩니다.

```bash
git init
git add .
git commit -m "Deploy CABT Online Arena"
git branch -M main
git remote add origin 본인_GITHUB_저장소_URL
git push -u origin main
```

## 2. Railway 프로젝트 생성

1. Railway에 GitHub 계정으로 로그인합니다.
2. `New Project`를 누릅니다.
3. `Deploy from GitHub repo`를 선택합니다.
4. 방금 올린 저장소를 선택합니다.
5. Railway가 `Dockerfile`과 `railway.json`을 읽어 자동 빌드합니다.

첫 빌드는 카드 이미지와 AI가 포함되어 있어 시간이 조금 걸릴 수 있습니다.

## 3. 관리자 토큰 등록

서비스를 클릭하고 `Variables`에서 다음 변수를 추가합니다.

```text
CABT_ADMIN_TOKEN=충분히_긴_임의_문자열
```

32자 이상을 권장합니다. 예시는 PowerShell에서 만들 수 있습니다.

```powershell
-join ((48..57)+(65..90)+(97..122) | Get-Random -Count 48 | ForEach-Object {[char]$_})
```

나머지 무료 플랜 권장값은 Dockerfile에 이미 들어 있습니다. 필요할 때만 `RAILWAY_VARIABLES.txt`를 참고해 덮어쓰면 됩니다.

## 4. `/data` 영구 Volume 연결

1. CABT 서비스를 선택합니다.
2. `Settings`의 `Volumes` 또는 프로젝트의 `Add Volume`을 선택합니다.
3. 서비스에 연결합니다.
4. **Mount Path를 정확히 `/data`**로 설정합니다.
5. 재배포합니다.

보관 위치:

```text
/data/records
/data/submissions
/data/user_data
```

Volume 없이도 서버는 실행되지만 재배포·재시작 때 결과가 사라질 수 있습니다.

## 5. 공개 주소 생성

1. 서비스 `Settings`를 엽니다.
2. `Networking → Public Networking`으로 이동합니다.
3. `Generate Domain`을 누릅니다.
4. 생성된 `https://...up.railway.app` 주소를 엽니다.

상태 확인 주소:

```text
https://본인주소/api/health
```

정상 예시:

```json
{
  "ok": true,
  "version": "5.1.0-railway",
  "public_mode": true,
  "online_matching": true,
  "workers": {"active": 0, "limit": 1}
}
```

## 6. 관리자 화면

```text
https://본인주소/admin
```

Variables에 넣은 `CABT_ADMIN_TOKEN`을 입력하면 제출된 결과와 기록 ZIP을 확인할 수 있습니다. 토큰은 다른 사용자에게 공개하지 마세요.

## 7. 온라인 대전 사용

- 빠른 매칭: 두 사용자가 각각 이름과 60장 덱을 선택해 빠른 매칭을 누릅니다.
- 친구 방: 한 명이 방을 만들고 6자리 코드를 상대에게 전달합니다.
- 같은 브라우저 쿠키가 유지되면 새로고침 후 진행 중 게임으로 복귀합니다.
- 배포나 서버 재시작이 발생하면 진행 중 게임은 복구되지 않습니다.

## 무료 플랜에서의 동시 게임 제한

무료 플랜 기본 설정:

```text
CABT_MAX_ACTIVE_WORKERS=1
CABT_MAX_SESSIONS=1
CABT_MAX_PVP_MATCHES=1
```

CABT 서버와 PvP 워커 하나를 실행했을 때 메모리 사용량이 약 350MB 수준이므로, 0.5GB RAM에서는 전체 워커 1개가 안전한 기본값입니다. AI 게임과 PvP 게임을 동시에 시작하면 두 번째 요청은 한도 메시지를 받습니다.

Hobby 이상 예시:

```text
CABT_MAX_ACTIVE_WORKERS=3
CABT_MAX_SESSIONS=3
CABT_MAX_PVP_MATCHES=3
```

실제 Railway Metrics에서 메모리를 확인하면서 한 단계씩 늘리세요.

## 업데이트 배포

GitHub의 `main` 브랜치에 새 커밋을 푸시하면 Railway가 자동 재배포합니다. Volume이 연결되어 있으면 완료된 기록은 유지됩니다. Volume 서비스는 새 배포와 이전 배포를 동시에 마운트할 수 없으므로 업데이트 중 잠깐의 중단이 생길 수 있습니다.

## 자주 발생하는 오류

### Application failed to respond

- Railway가 주입한 `PORT`를 서버가 사용해야 합니다. 이 패키지는 자동 대응합니다.
- `Dockerfile`과 `app/server.py`가 저장소 루트 기준 올바른 위치인지 확인합니다.
- Deploy Logs에서 Python 예외를 확인합니다.

### Healthcheck failed

- `/api/health`가 200을 반환하는지 Deploy Logs에서 확인합니다.
- 첫 빌드가 아니라 실행 단계에서 실패했다면 누락된 파일이 없는지 확인합니다.

### 기록이 재배포 후 사라짐

- Volume Mount Path가 `/data`인지 확인합니다.
- `CABT_DATA_DIR`을 다른 경로로 바꾸지 않았는지 확인합니다.

### 두 번째 게임이 시작되지 않음

- 무료 플랜 기본 동작입니다. 현재 워커가 종료될 때까지 기다리거나, Variables에서 활성 워커 수를 늘리려면 상위 플랜의 RAM을 사용하세요.
- AI 게임은 대기실로 돌아가며 종료할 때 즉시 워커 슬롯을 반환하도록 수정되어 있습니다.

### 배포 후 진행 중 게임이 끊김

- 현재 매칭과 네이티브 엔진 상태는 메모리에 있습니다. 배포·재시작 시 진행 중 게임은 종료되는 것이 정상입니다. 완료된 기록만 Volume에 유지됩니다.

## 보안 주의

공개 모드에서는 사용자 AI ZIP 업로드와 서버 공용 덱 저장·삭제가 차단됩니다. `app/agents/`에 포함할 AI는 신뢰할 수 있는 코드만 사용하세요. AI Python 코드는 서버 프로세스 권한으로 실행됩니다.
