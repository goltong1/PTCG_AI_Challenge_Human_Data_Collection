# GitHub Pages 배포 방법

## 1. 저장소 만들기

GitHub에서 새 공개 저장소를 만듭니다. 예: `cabt-local-arena`

## 2. Pages 파일 업로드

이 ZIP의 내용을 풀고 다음 파일과 폴더가 저장소 최상단에 오도록 업로드합니다.

```text
index.html
404.html
.nojekyll
app.js
style.css
assets/
README.md
DEPLOY_GITHUB_PAGES.md
```

`index.html`이 한 단계 아래 폴더에 들어가면 안 됩니다.

## 3. Windows 프로그램을 Release에 올리기

저장소 오른쪽의 **Releases → Create a new release**로 이동합니다.

- Tag: `v6.0.0`
- Release title: `CABT Local AI Arena v6`
- Asset: `CABT_Local_AI_Arena_v6.zip`

파일 이름을 정확히 유지해야 Pages의 다운로드 버튼이 자동으로 최신 Release Asset을 가리킵니다.

## 4. GitHub Pages 활성화

저장소에서 다음 순서로 설정합니다.

```text
Settings
→ Pages
→ Build and deployment
→ Source: Deploy from a branch
→ Branch: main
→ Folder: / (root)
→ Save
```

배포가 완료되면 다음 형태의 주소가 생성됩니다.

```text
https://사용자명.github.io/저장소명/
```

## 5. 업데이트

정적 사이트를 수정할 때는 `main` 브랜치 파일을 교체하면 됩니다. Windows 프로그램을 갱신할 때는 새 Release를 만들고 Asset 이름을 계속 `CABT_Local_AI_Arena_v6.zip`으로 유지하세요.

## 개인정보와 저장 위치

- 리플레이 파일은 브라우저에서 직접 읽습니다.
- 사이트 서버로 파일을 전송하는 API가 없습니다.
- 최근 리플레이는 해당 브라우저의 IndexedDB에만 저장됩니다.
- 브라우저 사이트 데이터 삭제 시 IndexedDB 기록도 삭제됩니다.
