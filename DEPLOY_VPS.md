# CABT Online Arena · 일반 VPS 배포

고정 월 비용과 서버 제어권을 우선하면 **Ubuntu VPS + Docker Compose + Caddy** 구성이 가장 단순합니다. 이 폴더에는 `docker-compose.vps.yml`, `Caddyfile`, `.env.example`이 포함되어 있습니다.

## 권장 사양

- Ubuntu 24.04 LTS
- 2 vCPU 권장
- RAM 2GB 최소, 동시 대전이 많으면 4GB 이상
- SSD 20GB 이상
- 한국과 가까운 도쿄 또는 싱가포르 리전
- 공인 IPv4 또는 IPv6와 도메인

## 1. DNS 설정

사용할 도메인의 A 레코드를 VPS IPv4 주소로 연결합니다.

```text
arena.example.com -> VPS 공인 IP
```

## 2. Docker 설치

Ubuntu 서버에서 Docker Engine과 Compose 플러그인을 설치합니다. 설치 후 아래 명령이 동작해야 합니다.

```bash
docker --version
docker compose version
```

## 3. 프로젝트 업로드

Git으로 가져오거나 SFTP로 프로젝트 폴더를 서버에 복사합니다.

```bash
git clone <YOUR_REPOSITORY_URL> cabt-online-arena
cd cabt-online-arena
```

## 4. 환경 변수 작성

```bash
cp .env.example .env
nano .env
```

반드시 다음 값을 변경합니다.

```dotenv
CABT_DOMAIN=arena.example.com
CABT_ADMIN_TOKEN=충분히-긴-임의-문자열
```

## 5. 실행

```bash
docker compose -f docker-compose.vps.yml up -d --build
docker compose -f docker-compose.vps.yml ps
```

Caddy가 도메인에 맞는 HTTPS 인증서를 자동으로 발급하고 CABT 컨테이너로 전달합니다.

로그 확인:

```bash
docker compose -f docker-compose.vps.yml logs -f --tail=200
```

업데이트:

```bash
git pull
docker compose -f docker-compose.vps.yml up -d --build
```

종료:

```bash
docker compose -f docker-compose.vps.yml down
```

`down -v`는 기록 Volume까지 삭제하므로 사용하지 마세요.

## 방화벽

외부에는 80/TCP, 443/TCP, 443/UDP만 엽니다. CABT의 8765 포트는 Compose 내부 네트워크에서만 사용합니다.

## 백업

CABT 데이터는 Docker의 `cabt-data` Volume에 저장됩니다. 정기적으로 해당 Volume을 압축 백업하거나 호스트 디렉터리 bind mount로 바꿔 별도 백업 정책을 적용하세요.

## 현재 버전의 확장 제한

현재 매칭 대기열과 활성 게임은 메모리 기반이므로 컨테이너를 한 개만 실행해야 합니다. 여러 서버로 확장하려면 Redis, 공유 영구 저장소, 게임별 고정 라우팅이 필요합니다.
