# Study Progress

Created: 2026-05-31

## Understood

1. Overall DQN flow
   - DQN is Q-learning with a neural network, replay buffer, and target network.

2. Q-value and Bellman target
   - `Q(s, a)` is the expected future return after taking action `a` in state `s`.
   - Bellman target is a bootstrapped training target, not a true supervised label.
   - `target_q_net` estimates the future-value part of the target.
   - Backprop updates `q_net`, not `target_q_net`.
   - Periodically, `q_net` is copied into `target_q_net`.

3. Value function vs Q function
   - `V(s)` is the value of a state.
   - `Q(s, a)` is the value of taking a specific action in a state.
   - `V*(s) = max_a Q*(s, a)`.

4. Replay buffer
   - Stores actual transitions `(s, a, r, s', done)`.
   - Samples random minibatches for DQN updates.
   - Does not invent actions during sampling.

## Next

1. Understand `QNetwork`
   - Why the input is state only.
   - Why the output has one Q-value per action.
   - How `gather` selects the Q-value for the action stored in replay buffer.

2. Understand epsilon-greedy action selection
   - How the agent chooses between random exploration and greedy exploitation.
   - Why this is the key ablation variable in the project.

