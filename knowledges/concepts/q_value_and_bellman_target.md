# Q-value And Bellman Target

Created: 2026-05-31

## Why This Matters

DQN을 이해하려면 결국 이 질문에 답할 수 있어야 한다.

> Q-network는 무엇을 예측하고, 그 예측값을 어떤 정답에 맞추는가?

답은 다음 두 가지다.

- 예측값: `Q(s, a)`
- 학습 target: `r + gamma * max_a' Q_target(s', a')`

## Return

강화학습에서 return은 지금부터 앞으로 받을 reward의 합이다.
미래 reward는 할인해서 더한다.

```text
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

`gamma`는 discount factor다.

- `gamma`가 0에 가까우면 당장의 reward를 거의 전부로 본다.
- `gamma`가 1에 가까우면 먼 미래 reward도 중요하게 본다.

이번 과제 기본값은 `gamma = 0.99`다.
MountainCar에서는 목표에 빨리 도달해야 하므로 미래까지 꽤 길게 보는 설정이다.

## Q-value

Q-value는 state-action pair의 가치다.

```text
Q(s, a) = state s에서 action a를 했을 때 기대되는 discounted future return
```

MountainCar에서는 reward가 대체로 매 step `-1`이다.
따라서 Q-value도 보통 음수다.

```text
Q = -200  -> 앞으로 대략 200 step 걸릴 것 같은 나쁜 행동
Q = -120  -> 앞으로 대략 120 step 걸릴 것 같은 더 좋은 행동
```

이 환경에서는 "Q-value가 크다"는 말이 "0에 더 가깝다"는 뜻이다.
즉 `-120`은 `-200`보다 좋은 값이다.

## Q-network Output

DQN의 network는 action 하나의 Q-value만 뱉는 것이 아니라, 모든 action의 Q-value를 한 번에 뱉는다.

```text
input:
    state = [position, velocity]

output:
    [Q(s, left), Q(s, no_push), Q(s, right)]
```

예를 들어 출력이 다음과 같다고 하자.

```text
[-180, -175, -160]
```

그러면 가장 큰 값은 `-160`이고, 이 action이 현재 network가 보기에 가장 좋은 action이다.

```text
argmax([-180, -175, -160]) = right
```

## Value Function vs Q Function

`V(s)`와 `Q(s, a)`는 다르다.

```text
V(s):
    state s 자체의 가치
    state s에서 시작하면 앞으로 받을 expected return

Q(s, a):
    state s에서 action a를 먼저 했을 때의 가치
    그 action을 한 뒤 앞으로 받을 expected return
```

관계는 다음과 같다.

```text
V*(s) = max_a Q*(s, a)
```

즉 optimal value function은 그 state에서 가능한 action 중 가장 좋은 Q-value와 같다.

DQN은 `V(s)`를 직접 학습하지 않고 `Q(s, a)`를 학습한다.
이유는 action을 고르려면 action별 가치를 비교해야 하기 때문이다.

이번 과제의 Q-network 구현은 action을 input으로 받지 않는다.
대신 state만 input으로 받고 모든 action의 Q-value를 한 번에 출력한다.

```text
q_net(state) -> [Q(s, left), Q(s, no_push), Q(s, right)]
```

특정 action `a`의 Q-value가 필요하면 output vector에서 해당 action index를 고른다.
PyTorch 구현에서는 보통 `gather`를 쓴다.

```text
q_net(states):                    (batch_size, 3)
actions:                          (batch_size, 1)
q_net(states).gather(1, actions): (batch_size, 1)
```

## The Bellman Idea

Q-value는 한 번에 전체 미래를 다 알 수 없으므로, 한 step 뒤의 문제로 나누어 생각한다.

현재 transition이 다음과 같다고 하자.

```text
s  = current state
a  = chosen action
r  = observed reward
s' = next state
```

그러면 현재 행동의 가치는 다음과 같이 볼 수 있다.

```text
Q(s, a) ≈ r + gamma * best_future_value_from_s'
```

다음 state `s'`에 도착한 뒤에는 가장 좋은 action을 고른다고 가정한다.

```text
best_future_value_from_s' = max_a' Q(s', a')
```

그래서 Bellman optimality equation은 다음 형태가 된다.

```text
Q*(s, a) = E[ r + gamma * max_a' Q*(s', a') ]
```

## Bellman Target In DQN

DQN에서는 이 식을 학습 target으로 사용한다.

```text
target = r + gamma * max_a' Q_target(s', a')
```

하지만 episode가 끝난 terminal transition이면 다음 state 이후의 미래는 없다.
그래서 bootstrapping을 제거한다.

```text
if done:
    target = r
else:
    target = r + gamma * max_a' Q_target(s', a')
```

코드에서는 보통 이렇게 쓴다.

```text
target = rewards + gamma * (1 - dones) * next_q_values
```

여기서 `dones`가 1이면 미래항이 0이 된다.

## Important: How Do We Know The Target?

정확한 true target은 모른다.
그걸 모르기 때문에 Q-network를 학습하는 것이 맞다.

DQN의 `target`은 supervised learning의 완전한 label과 다르다.
환경에서 실제로 관측한 부분과 network가 추정한 부분을 섞어서 만든다.

```text
target = observed immediate reward + estimated future value
```

더 구체적으로:

```text
known from environment:
    r
    s'
    done

estimated by target network:
    max_a' Q_target(s', a')
```

그래서 Bellman target은 "진짜 정답"이라기보다 현재 우리가 만들 수 있는 가장 일관된 학습 목표다.
이런 방식을 bootstrapping이라고 한다.
자기 자신의 추정값을 이용해 더 나은 추정값으로 조금씩 고쳐간다는 뜻이다.

Q-learning/DQN은 이 과정을 많은 transition에 반복하면서 Bellman equation의 fixed point에 가까워지려 한다.

## Important: Which Network Gets Backprop?

Bellman target을 숫자로 계산한 뒤, 그 숫자에 대해 backprop을 하는 것은 맞다.
하지만 backprop으로 업데이트되는 network는 `target_q_net`이 아니라 `q_net`이다.

```text
1. target_q_net으로 Bellman target 숫자를 계산한다.
2. q_net으로 현재 prediction Q(s, a)를 계산한다.
3. loss = MSE(prediction, target)을 계산한다.
4. optimizer.step()으로 q_net만 업데이트한다.
5. 몇 step 뒤 q_net의 parameter를 target_q_net에 복사한다.
```

코드 관점:

```python
with torch.no_grad():
    next_q_values = target_q_net(next_states).max(dim=1, keepdim=True).values
    targets = rewards + gamma * (1 - dones) * next_q_values

current_q_values = q_net(states).gather(1, actions)
loss = F.mse_loss(current_q_values, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

여기서 `torch.no_grad()` 때문에 target 계산 쪽으로는 gradient가 흐르지 않는다.
즉 `target_q_net`은 이 loss로 직접 학습되지 않는다.

일정 step 뒤에만 다음 복사가 일어난다.

```python
target_q_net.load_state_dict(q_net.state_dict())
```

정리:

```text
Bellman target을 만든다:
    target_q_net 사용, gradient 없음

prediction을 만든다:
    q_net 사용, gradient 있음

loss로 학습한다:
    q_net만 업데이트

target network를 갱신한다:
    q_net을 target_q_net에 복사
```

## Prediction vs Target

DQN loss는 current Q-network의 예측값과 Bellman target 사이의 차이다.

예측값:

```text
prediction = Q_online(s, a)
```

target:

```text
target = r + gamma * (1 - done) * max_a' Q_target(s', a')
```

loss:

```text
loss = MSE(prediction, target)
```

즉 DQN은 다음을 반복한다.

```text
Q_online(s, a)를 Bellman target에 가까워지도록 업데이트한다.
```

중요한 점:

```text
비교하는 prediction은 max_a Q(s, a)가 아니다.
replay buffer에 저장된 실제 action a에 대한 Q(s, a)다.
```

즉 transition이 `(s, a, r, s', done)`이면:

```text
prediction = Q_online(s, a)
target     = r + gamma * (1 - done) * max_a' Q_target(s', a')
```

현재 state `s`에서는 저장된 action `a`의 Q-value만 업데이트한다.
반면 next state `s'`에서는 target을 만들기 위해 가능한 action 중 최대 Q-value를 사용한다.

## Why Target Network Is Used

Bellman target도 Q-network의 예측값을 사용한다.
그런데 같은 network를 prediction과 target 양쪽에 쓰면 target이 계속 같이 흔들린다.

그래서 DQN은 두 network를 둔다.

```text
q_net:
    학습되는 online network

target_q_net:
    Bellman target 계산에 쓰는 느리게 갱신되는 network
```

수식으로 쓰면 다음과 같다.

```text
prediction = Q(s, a; theta)
target     = r + gamma * (1 - done) * max_a' Q(s', a'; theta_minus)
loss       = MSE(prediction, target)
```

여기서:

```text
theta:
    q_net의 parameter
    optimizer가 매 update step마다 바꾼다

theta_minus:
    target_q_net의 parameter
    gradient descent로 매번 바뀌지 않는다
    일정 주기마다 theta를 복사해서 갱신한다
```

과제 코드에서는 일정 step마다 다음을 수행해야 한다.

```python
target_q_net.load_state_dict(q_net.state_dict())
```

즉 `target_q_net`은 독립적으로 따로 학습되는 network가 아니다.
`q_net`의 조금 오래된 snapshot이다.

이렇게 하는 이유는 target 계산식 자체가 network 예측값을 포함하기 때문이다.
prediction을 만드는 network와 target을 만드는 network가 매 gradient step마다 동시에 바뀌면, 학습 목표가 너무 빨리 움직여 불안정해질 수 있다.
target network를 잠시 고정해 두면 q_net이 비교적 안정적인 목표를 따라갈 수 있다.

## Clarification: `target` vs `Q_target`

용어가 비슷해서 헷갈리기 쉽다.

```text
target = r + gamma * max_a' Q_target(s', a')
```

여기서 왼쪽의 `target`은 학습 정답 역할을 하는 scalar/tensor 값이다.
보통 Bellman target 또는 TD target이라고 부른다.

오른쪽의 `Q_target`은 target network가 예측한 Q-value다.
즉 `target_q_net(next_states)`에서 나온 값이다.

중요한 차이:

```text
target:
    loss 계산에 쓰는 정답 값
    gradient descent로 직접 업데이트되는 parameter가 아니다

Q_target / target_q_net:
    Bellman target을 계산하기 위해 사용하는 neural network
    매 gradient step마다 학습되는 것이 아니라, 일정 주기마다 q_net에서 복사된다
```

이번 과제의 기본 설정에서는 `target_update_freq = 400`이므로, 보통 400 environment step마다 다음 복사를 한다.

```python
target_q_net.load_state_dict(q_net.state_dict())
```

## Clarification: What `done` Means

`done`은 "어떤 state 자체의 이름"이라기보다 transition의 종료 flag다.

```text
(s, a, r, s', done)
```

이 뜻은:

```text
state s에서 action a를 했더니 reward r을 받고 next_state s'로 갔다.
그리고 그 결과 episode가 끝났으면 done = True.
```

`done = True`이면 `s'` 이후의 미래 reward를 더하면 안 된다.
그래서 Bellman target에서 future value term을 제거한다.

```text
target = r + gamma * (1 - done) * max_a' Q_target(s', a')
```

`done = 1`이면:

```text
target = r
```

`done = 0`이면:

```text
target = r + gamma * max_a' Q_target(s', a')
```

Gymnasium에서는 episode 종료가 `terminated`와 `truncated`로 나뉜다.
과제 helper인 `step_env`는 둘을 합쳐서 `done = terminated or truncated`로 만든다.

## How This Maps To `compute_dqn_loss`

과제 TODO는 이 순서로 구현된다.

1. `q_net(states)`로 모든 action의 Q-value를 구한다.
2. `gather`로 실제 replay buffer에 저장된 action의 Q-value만 고른다.
3. `target_q_net(next_states)`로 next state의 action별 Q-value를 구한다.
4. `max(dim=1)`로 next state에서 가장 좋은 Q-value를 고른다.
5. `rewards + gamma * (1 - dones) * next_q_values`로 target을 만든다.
6. `F.mse_loss(current_q_values, targets)`를 반환한다.

Shape 감각:

```text
states:      (batch_size, 2)
actions:     (batch_size, 1)
rewards:     (batch_size, 1)
next_states: (batch_size, 2)
dones:       (batch_size, 1)

q_net(states):               (batch_size, 3)
current_q_values via gather: (batch_size, 1)
target_q_net(next_states):   (batch_size, 3)
next_q_values via max:       (batch_size, 1)
targets:                    (batch_size, 1)
```

## One Concrete Example

Assume:

```text
reward = -1
gamma = 0.99
done = False
max_a' Q_target(s', a') = -120
```

Then:

```text
target = -1 + 0.99 * (-120)
       = -119.8
```

If the current network predicts:

```text
Q_online(s, a) = -150
```

then the network is too pessimistic for that transition.
Gradient descent will push `Q_online(s, a)` upward toward `-119.8`.

If `done = True`, then:

```text
target = reward = -1
```

There is no future value after a terminal transition.
