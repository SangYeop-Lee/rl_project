# DQN Learning Path

Created: 2026-05-23

## Goal

EE619 term project를 풀기 위해 DQN을 "구현 가능한 수준"으로 이해한다.
최종 목표는 MountainCar-v0에서 두 epsilon-greedy exploration schedule을 공정하게 비교하는 것이다.

## What We Need To Understand First

1. Reinforcement learning problem setup
   - state, action, reward, episode, return
   - MountainCar-v0에서 state/action/reward가 무엇인지

2. Q-value
   - Q(s, a)가 의미하는 것
   - "현재 상태 s에서 행동 a를 했을 때 앞으로 받을 누적 보상 기대값"으로 이해한다.
   - 좋은 policy는 보통 argmax_a Q(s, a)를 고른다.

3. Q-learning target
   - 핵심 업데이트 목표:
     `target = reward + gamma * max_a' Q(next_state, a')`
   - terminal transition에서는 future value를 더하지 않는다:
     `target = reward`

4. Why DQN uses a neural network
   - MountainCar state는 연속값 position, velocity이다.
   - 모든 state-action 값을 표로 저장하기 어렵기 때문에 neural network가 Q(s, a)를 근사한다.
   - 입력: `[position, velocity]`
   - 출력: 세 action에 대한 Q-value `[Q_left, Q_stay, Q_right]`

5. Replay buffer
   - 최근 transition `(s, a, r, s', done)`을 저장한다.
   - 학습할 때 순서대로 바로 쓰지 않고 무작위 minibatch를 뽑는다.
   - 이유: 연속된 경험의 강한 상관을 줄이고, 과거 경험을 재사용하기 위해서.

6. Target network
   - DQN은 target 계산에도 Q-network를 쓴다.
   - 같은 network를 계속 움직이는 target으로 쓰면 학습이 불안정해진다.
   - 그래서 천천히 업데이트되는 `target_q_net`을 따로 둔다.

7. Epsilon-greedy exploration
   - 확률 epsilon으로 random action을 고른다.
   - 확률 1 - epsilon으로 현재 Q-network가 가장 좋다고 보는 action을 고른다.
   - 이번 과제의 핵심 비교 변수:
     - less_exploration: epsilon이 빨리 줄고 0.01로 수렴
     - more_exploration: epsilon이 천천히 줄고 0.05로 수렴

8. Training loop
   - 환경에서 행동한다.
   - transition을 replay buffer에 저장한다.
   - warmup 이후 minibatch를 뽑아 DQN loss로 업데이트한다.
   - 일정 주기마다 target network를 갱신한다.
   - episode return, loss, epsilon, success 여부를 기록한다.

## Project TODOs Mapped To Concepts

1. `ReplayBuffer`
   - transition 저장과 minibatch sampling 이해

2. `QNetwork`
   - state를 action별 Q-value로 바꾸는 MLP 이해

3. `select_action`
   - epsilon-greedy policy 이해

4. `compute_dqn_loss`
   - Bellman target과 MSE loss 이해

5. `dqn_update`
   - gradient descent 한 step 이해

6. `train_dqn`
   - 전체 학습 루프와 실험 기록 이해

## Our Study Order

1. MountainCar 문제 자체 이해
2. Q-value와 Bellman equation 이해
3. DQN 구성요소 4개 이해: Q-network, replay buffer, target network, epsilon-greedy
4. `project.py` TODO를 개념과 연결해서 구현
5. 짧은 디버그 실험으로 코드 검증
6. 전체 6개 실험 실행
7. plots/tables를 해석하고 report discussion 작성

