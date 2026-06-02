# Epsilon-Greedy

Created: 2026-06-01

## Core Idea

Epsilon-greedy는 exploration과 exploitation을 섞는 action selection rule이다.

```text
with probability epsilon:
    choose a random action

with probability 1 - epsilon:
    choose argmax_a Q(s, a)
```

즉 epsilon은 "일부러 모르는 행동을 해볼 확률"이다.

## Why We Need It

초기 `q_net`은 아직 학습되지 않았기 때문에 Q-value가 믿을 만하지 않다.
처음부터 `argmax Q(s, a)`만 따르면 잘못된 action을 계속 반복할 수 있다.

Epsilon-greedy는 이런 문제를 줄인다.

```text
early training:
    high epsilon -> many random actions -> more exploration

late training:
    low epsilon -> mostly greedy actions -> more exploitation
```

## Action Selection In This Project

과제의 `select_action`은 다음 로직을 구현해야 한다.

```python
if random.random() < epsilon:
    return random action
else:
    return argmax q_net(state)
```

MountainCar의 action 개수는 3개다.

```text
action_dim = 3
possible actions = 0, 1, 2
```

random action은 다음 중 하나를 고른다.

```text
0 = push left
1 = no push
2 = push right
```

greedy action은 q_net이 출력한 Q-values 중 가장 큰 index다.

```text
q_net(state) = [-180, -175, -160]
argmax = 2
```

## Why `torch.no_grad()` Is Used

Action selection에서는 network를 학습하지 않는다.
그저 현재 state에서 Q-value를 계산해 action을 고를 뿐이다.

그래서 greedy branch에서는 gradient를 추적하지 않는 것이 맞다.

```python
with torch.no_grad():
    q_values = q_net(state_tensor)
    action = torch.argmax(q_values).item()
```

Backprop은 `dqn_update`에서 loss를 계산할 때만 한다.

## Epsilon Schedule

이번 과제는 epsilon을 고정하지 않고 step에 따라 줄인다.

```text
epsilon(step) = eps_end + (eps_start - eps_end) * exp(-step / decay_steps)
```

처음에는 `eps_start = 1.0`이다.
즉 거의 완전 random behavior로 시작한다.

시간이 지날수록 epsilon은 `eps_end`에 가까워진다.

## Required Schedules

과제는 정확히 두 schedule을 비교한다.

```text
less_exploration:
    eps_start = 1.0
    eps_end = 0.01
    decay_steps = 1,000

more_exploration:
    eps_start = 1.0
    eps_end = 0.05
    decay_steps = 5,000
```

해석:

```text
less_exploration:
    epsilon이 빨리 낮아진다.
    더 빨리 exploitation에 들어간다.

more_exploration:
    epsilon이 더 오래 높게 유지된다.
    더 오래 다양한 transition을 수집한다.
```

## Why This Affects Learning

Epsilon-greedy는 단지 action만 바꾸는 것이 아니다.
Replay buffer에 들어가는 데이터 분포를 바꾼다.

```text
high epsilon:
    many random transitions
    broader state/action coverage

low epsilon:
    many q_net-driven transitions
    more exploitation of current learned behavior
```

따라서 exploration schedule은 DQN이 어떤 경험을 replay buffer에 저장하고, 어떤 데이터로 학습하는지를 바꾼다.

## Project Interpretation Questions

실험 결과를 해석할 때 볼 질문:

1. More exploration이 평균 final return을 높였는가?
2. Less exploration이 더 빠르게 좋아졌지만 불안정했는가?
3. 어떤 schedule이 seed 간 variance가 작았는가?
4. Reward curve와 loss curve가 같은 결론을 주는가?
5. MountainCar에서 momentum을 발견하는 데 exploration이 어떤 역할을 했는가?

