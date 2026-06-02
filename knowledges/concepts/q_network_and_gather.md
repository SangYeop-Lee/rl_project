# QNetwork And Gather

Created: 2026-05-31

## What QNetwork Represents

DQN의 Q-network는 Q-function을 neural network로 근사한 것이다.

개념적으로는 다음을 배우고 싶다.

```text
Q(s, a)
```

즉 state `s`에서 action `a`를 했을 때의 expected future return이다.

하지만 이번 과제 구현에서는 action을 input으로 넣지 않는다.
state만 input으로 넣고, 가능한 모든 action의 Q-value를 한 번에 출력한다.

```text
q_net(s) -> [Q(s, left), Q(s, no_push), Q(s, right)]
```

MountainCar-v0에서는:

```text
state_dim = 2
action_dim = 3
```

그래서 network의 입출력 shape은 다음과 같다.

```text
input:  (batch_size, 2)
output: (batch_size, 3)
```

## Why Output All Actions At Once?

action별 Q-value를 한 번에 출력하면 action selection이 쉽다.

```text
q_values = q_net(state)
action = argmax(q_values)
```

예시:

```text
q_net(state) = [-180, -175, -160]
```

가장 큰 값은 `-160`이므로 action index 2를 고른다.

```text
action = 2  # push right
```

## QNetwork Architecture In This Project

과제에서 제안한 architecture:

```text
Linear(state_dim, hidden_dim)
ReLU
Linear(hidden_dim, hidden_dim)
ReLU
Linear(hidden_dim, action_dim)
```

MountainCar 기준으로는:

```text
Linear(2, 64)
ReLU
Linear(64, 64)
ReLU
Linear(64, 3)
```

마지막 layer 뒤에는 ReLU를 붙이지 않는다.
Q-value는 음수도 나올 수 있기 때문이다.
MountainCar에서는 reward가 대부분 `-1`이어서 Q-value가 음수인 경우가 자연스럽다.

## Why `gather` Is Needed

Replay buffer에서 batch를 뽑으면 action도 같이 나온다.

```text
states:  (batch_size, 2)
actions: (batch_size, 1)
```

`q_net(states)`는 모든 action의 Q-value를 준다.

```text
q_net(states): (batch_size, 3)
```

하지만 loss에서 비교해야 하는 prediction은 모든 action이 아니다.
Replay buffer에 저장된 실제 action `a`에 대한 Q-value만 비교한다.

```text
prediction = Q(s, a)
```

그래서 `gather`로 각 row에서 해당 action column만 뽑는다.

## Concrete Batch Example

Assume batch size is 3.

```text
q_net(states) =
[
  [-180, -175, -160],
  [-130, -150, -140],
  [-200, -190, -195],
]

actions =
[
  [2],
  [0],
  [1],
]
```

그러면:

```text
q_net(states).gather(1, actions) =
[
  [-160],
  [-130],
  [-190],
]
```

각 transition에서 실제로 선택했던 action의 Q-value만 뽑힌다.

이 값이 Bellman target과 비교되는 prediction이다.

```text
loss = MSE(Q_online(s, a), Bellman target)
```

## Difference Between Action Selection And Training Loss

Action selection에서는 모든 action Q-value 중 가장 큰 값을 고른다.

```text
action = argmax_a Q(s, a)
```

Training loss에서는 replay buffer에 저장된 실제 action의 Q-value만 업데이트한다.

```text
prediction = Q(s, stored_action)
```

Target 계산에서는 next state의 action 중 최대 Q-value를 쓴다.

```text
target = r + gamma * (1 - done) * max_a' Q_target(s', a')
```

이 세 가지를 구분해야 한다.

