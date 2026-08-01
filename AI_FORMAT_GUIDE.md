# CABT AI ZIP 호환 규격

## 필수 파일

- `main.py`
- `deck.csv` — 정수 카드 ID 60개
- `cg/api.py`
- `cg/game.py`
- Windows 실행용 `cg/cg.dll`

Kaggle 제출 ZIP처럼 루트에 바로 있어도 되고, 하나 이상의 상위 폴더로 감싸져 있어도 됩니다. 로더는 `main.py`와 `deck.csv`가 함께 있는 가장 적합한 제출 루트를 찾습니다.

## 진입점 계약

```python
def agent(observation: dict) -> list[int]:
    if observation.get("select") is None:
        return MY_60_CARD_DECK
    return [SELECTED_OPTION_INDEX]
```

게임 중 반환 배열은 다음 조건을 만족해야 합니다.

- 모든 값이 정수
- 중복 index 없음
- 각 index는 `0 <= index < len(observation["select"]["option"])`
- 배열 길이는 `minCount` 이상 `maxCount` 이하

## 경로 처리

GUI는 AI를 실행할 때 해당 제출 폴더를 현재 작업 폴더로 설정합니다. `deck.csv`를 상대 경로로 여는 일반적인 Kaggle 제출 코드가 그대로 동작합니다.

## ZIP 안전 검사

- 절대 경로, `..` 경로, 심볼릭 링크 거부
- 압축 파일 최대 256MB
- 압축 해제 최대 1GB
- 최대 파일 수 20,000개

AI 코드는 샌드박스가 아니라 사용자의 로컬 프로세스에서 실행되므로 신뢰할 수 있는 제출만 추가해야 합니다.
