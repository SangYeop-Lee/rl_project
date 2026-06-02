# MountainCar And DQN Basics

Created: 2026-05-23

## MountainCar-v0 Problem

MountainCar는 언덕 사이에 있는 차가 오른쪽 목표 지점까지 올라가야 하는 문제다.
차의 엔진 힘만으로는 바로 오른쪽 언덕을 올라가기 어렵기 때문에, 왼쪽으로도 움직이며 속도를 모아야 한다.

이 점이 이 과제의 discussion question과 직접 연결된다:

> In MountainCar, why can moving away from the goal initially be useful?

답의 핵심은 "잠깐 목표 반대 방향으로 움직이는 것이 위치 에너지를 이용해 충분한 momentum을 만드는 데 도움이 된다"는 것이다.

## State

MountainCar의 state는 2차원이다.

```text
state = [position, velocity]
```

- `position`: 차의 현재 위치
- `velocity`: 차의 현재 속도

과제의 Q-network 입력 차원이 2인 이유가 이것이다.

```python
QNetwork(state_dim=2, action_dim=3)
```

## Action

MountainCar의 action은 3개다.

```text
0 = push left
1 = no push
2 = push right
```

DQN은 각 state에서 이 3개 action의 Q-value를 모두 예측한다.

```text
QNetwork([position, velocity])
    -> [Q(left), Q(no_push), Q(right)]
```

그리고 greedy policy는 가장 큰 Q-value를 가진 action을 선택한다.

```text
action = argmax_a Q(state, a)
```

## Reward And Return

MountainCar-v0는 보통 매 step마다 reward가 `-1`이다.
목표에 빨리 도달할수록 episode가 짧아지고, 총 return이 덜 나빠진다.

예시:

```text
200 steps 동안 실패 -> return = -200
150 steps 만에 성공 -> return = -150
110 steps 만에 성공 -> return = -110
```

따라서 이 환경에서는 return이 높다는 말이 "0에 더 가깝다"는 뜻이다.
`-110`은 `-200`보다 좋은 성능이다.

## Why Random Policy Is Weak

무작위 행동은 가끔 목표에 가까워질 수 있지만, momentum을 일관되게 만들기 어렵다.
그래서 random policy baseline은 보통 낮은 return을 낸다.

과제에서 random policy baseline mean/std를 요구하는 이유:

- DQN 결과가 random보다 나은지 비교할 기준이 필요하다.
- DQN이 실제로 학습했는지 확인할 수 있다.

## What DQN Learns

DQN은 직접 "오른쪽으로 가라" 같은 규칙을 배우는 것이 아니다.
각 state에서 action별 장기 가치를 예측하도록 학습한다.

```text
Q(s, a) = state s에서 action a를 했을 때 기대되는 discounted future return
```

MountainCar에서는 좋은 Q-function이 다음을 배워야 한다.

- 어떤 상태에서는 오른쪽으로 미는 것이 좋다.
- 어떤 상태에서는 왼쪽으로 미는 것이 나중에 더 좋은 결과를 만든다.
- 속도와 위치를 같이 봐야 한다.

즉, position만 보면 안 되고 velocity까지 함께 봐야 한다.

## Why Exploration Matters Here

초기 Q-network는 거의 아무것도 모른다.
처음부터 greedy action만 고르면 잘못된 Q-value에 갇힐 수 있다.

epsilon-greedy는 이 문제를 줄인다.

```text
with probability epsilon:
    choose random action
otherwise:
    choose argmax Q(state, action)
```

이번 과제의 핵심 비교:

```text
less_exploration:
    epsilon quickly decays to 0.01

more_exploration:
    epsilon decays more slowly and converges to 0.05
```

해석할 때 볼 질문:

- exploration을 더 오래 유지한 쪽이 더 안정적으로 goal-reaching behavior를 찾았는가?
- 빠르게 exploitation으로 간 쪽이 특정 seed에서 성능이 불안정했는가?
- reward curve와 loss curve가 같은 이야기를 하는가?

## Important Interpretation

Loss가 낮다고 항상 좋은 policy라는 뜻은 아니다.
DQN loss는 "현재 replay buffer에 있는 target에 Q-network가 얼마나 맞는가"를 재는 값이다.

하지만 좋은 policy의 기준은 결국 environment return이다.

그래서 report에서는 loss curve만 보면 안 되고 reward curve, final return table, success rate를 같이 봐야 한다.

