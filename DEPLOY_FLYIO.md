# CABT Online Arena · Fly.io 배포

이 프로젝트의 기본 권장 배포 대상은 **Fly.io 도쿄 리전(`nrt`)의 단일 Machine + 영구 Volume**입니다. 한국 사용자끼리 플레이할 때 지리적으로 가까우며, Docker 이미지 안의 네이티브 CABT 엔진과 게임별 자식 프로세스를 그대로 실행할 수 있습니다.

## 중요한 구조 제한

현재 온라인 매칭과 진행 중 게임은 서버 메모리에 있으며, 한 게임의 두 플레이어는 반드시 같은 서버 프로세스에 연결되어야 합니다. 따라서 이 버전은 다음 구성을 사용해야 합니다.

- Fly Machine: 정확히 1대
- 리전: `nrt` 1곳
- Volume: `/data`에 1개 연결
- 자동 정지: 끔

Machine을 여러 대로 늘리면 빠른 매칭 대기열과 게임 워커가 서로 분리됩니다. 다중 인스턴스로 확장하려면 Redis 기반 매칭 상태, 게임별 라우팅, 공유 기록 저장소를 추가해야 합니다.

## 1. 준비

1. 이 폴더를 GitHub 저장소에 올립니다.
2. Fly CLI(`flyctl`)를 설치하고 로그인합니다.
3. `fly.toml`의 첫 줄에 있는 앱 이름을 전 세계에서 유일한 소문자 이름으로 변경합니다.

예:

```toml
app = "ingyun-cabt-arena"
```

## 2. 앱과 Volume 생성

프로젝트 폴더에서 실행합니다.

```bash
fly auth login
fly apps create ingyun-cabt-arena
fly volumes create cabt_data --region nrt --size 10
```

`ingyun-cabt-arena` 부분은 `fly.toml`에 넣은 앱 이름과 같아야 합니다.

## 3. 관리자 토큰 등록

아래 예제 대신 충분히 긴 임의 문자열을 사용합니다.

```bash
fly secrets set CABT_ADMIN_TOKEN="replace-this-with-a-long-random-token"
```

관리자 페이지는 배포 후 `https://앱이름.fly.dev/admin`에서 열 수 있습니다.

## 4. 배포

```bash
fly deploy
fly scale count 1 --region nrt
fly status
```

로그 확인:

```bash
fly logs
```

상태 확인:

```text
https://앱이름.fly.dev/api/health
```

`online_matching: true`와 `pvp` 용량 정보가 표시되면 정상입니다.

## 5. 용량 조절

`fly.toml` 기본값은 2GB RAM, AI 세션 4개, 온라인 대전 4개입니다. 온라인 대전 하나마다 별도 네이티브 워커 프로세스가 생기므로 메모리 부족이 발생하면 먼저 동시 게임 수를 낮추거나 Machine 메모리를 늘리세요.

```toml
CABT_MAX_SESSIONS = "2"
CABT_MAX_PVP_MATCHES = "2"
CABT_MAX_PVP_WAITERS = "16"
```

Machine 크기 변경 예:

```bash
fly scale memory 4096
```

## 6. 데이터와 백업

영구 데이터는 `/data` 아래에 저장됩니다.

- `/data/records`: 원본 게임 기록
- `/data/submissions`: 사용자가 동의해 제출한 AI 대전 결과
- `/data/user_data`: 서버 데이터

Volume은 배포 중 유지되지만 단일 디스크 장애에 대한 완전한 복제 저장소는 아닙니다. 중요한 기록은 `/admin`에서 정기적으로 내려받거나 이후 S3 호환 오브젝트 스토리지로 이전하는 편이 안전합니다.

## 7. 업데이트

코드를 수정한 뒤 다시 실행합니다.

```bash
fly deploy
```

배포나 Machine 재시작 시 진행 중인 온라인 대전은 종료됩니다. 이미 저장된 기록 파일은 Volume에 남습니다.

## 문제 해결

### 매칭은 되지만 상대 화면이 갱신되지 않음

- 두 브라우저가 동일한 `*.fly.dev` 주소를 사용하는지 확인합니다.
- 프록시나 브라우저가 쿠키를 차단하지 않는지 확인합니다.
- `fly scale count 1 --region nrt`로 Machine이 정확히 1대인지 확인합니다.

### 서버가 자주 재시작됨

```bash
fly logs
fly status
```

메모리 부족 메시지가 있으면 동시 게임 한도를 낮추거나 RAM을 4GB로 올립니다.

### 관리자 페이지에 들어갈 수 없음

`CABT_ADMIN_TOKEN`이 등록되어 있는지 확인합니다.

```bash
fly secrets list
```
