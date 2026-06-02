# Experiments

이 디렉토리는 실험 기록을 남기는 곳이다.

기록할 내용:

1. 실험 이름
2. 날짜
3. 바꾼 조건
4. 고정한 조건
5. seeds
6. 주요 결과
7. 해석
8. 다음 액션

이번 과제의 required experiment는 다음 하나의 ablation이다.

- 비교 변수: epsilon-greedy exploration schedule
- schedule 1: `less_exploration`
- schedule 2: `more_exploration`
- seeds: `[0, 1, 2]`
- 고정 조건: DQN architecture, optimizer, replay buffer size, target update frequency, learning rate, discount factor, batch size, number of episodes

