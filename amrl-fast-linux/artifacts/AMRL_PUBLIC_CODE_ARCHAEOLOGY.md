# AMRL public-code archaeology and reconstruction boundary

Date: 2026-10-05.  This is a read-only audit of the public repositories and
their reachable Git history.  It is not an experimental result.

## Repositories and recoverability

| Repository | Public refs inspected | Result |
|---|---|---|
| `Albert-Dave/AMRL-Meta-Net` | `main`, all reachable commits, tags, releases, `colab_run.ipynb`, configs and scripts | One branch (`main`); no tags/releases; no unreachable Git objects.  The current code is a small online controller implementation. |
| `Albert-Dave/AMRL-FAST-META-NET` | `main`, all reachable commits, tags, releases, notebook, configs and scripts | One branch (`main`); no tags/releases; no unreachable Git objects.  Current `main` deletes all source files.  The only recoverable source snapshot is commit `9598ee8`; its code is incomplete and its directory names contain literal trailing spaces (for example `src /amrl_fast`). |
| `Albert-Dave/CQUPT_FYP_AI_Gym_Trainer_SSPA` | default branch tree | No files matching AMRL, meta-PPO, outer-loop, HalfCheetah or BSuite/Catch concepts. |

## Evidence table

`Paper says` records the manuscript protocol.  `Public code says` is limited
to executable or directly readable public code; documentation-only claims are
not treated as an implementation.

| Item | Paper says | Public code says | Found implementation | Status |
|---|---|---|---|---|
| Meta-controller dimensions | input 5, LSTM hidden 256, output 3 | `AMRL-Meta-Net`: input 6, hidden 64, three heads. `AMRL-FAST` historical snapshot: input 5, hidden 256, output 3. | `AMRL-FAST-META-NET@9598ee8:src /amrl_fast/{core.py,meta_learner.py}` | Partially found; Fast dimensions recoverable. |
| Adaptation frequency | 10,000 timesteps | Original `train_meta.py`: `t > 5000 and t % 5000 == 0`; CartPole/BSuite configs 5,000; MountainCar config 10,000. Fast historical default 10,000. | Original configs and Fast `core.py` | Conflicting public protocols; 10,000 only evidenced in Fast snapshot. |
| Outer state definition | Five controller diagnostics | Fast snapshot defines `[mean_reward, reward_std, gradient_norm, policy_entropy, value_loss]`; the last three values in its PPO wrapper are placeholders (1.0, 0.0, 0.0). Original code defines a different six-statistic state. | Fast `core.py`, `base_agents.py`; original `amrl/meta_state.py` | Feature names found, but operational diagnostics are not valid. |
| Outer action definition | Bounded adjustments to learning rate, gamma and exploration/entropy control | Fast snapshot outputs three `tanh` values.  It updates learning rate multiplicatively, gamma additively and PPO clip epsilon multiplicatively. | Fast `meta_learner.py`, `core.py` | Found, but action semantics conflict with manuscript wording (exploration/entropy versus PPO clip epsilon). |
| Outer reward definition | Reward improvement in the following adaptation window for the MI protocol | Fast snapshot uses current chunk mean reward minus prior chunk mean reward. Original `MetaReward` exists but is not consumed in training. | Fast `core.py`; original `amrl/meta_reward.py`, `train_meta.py` | One-step performance-difference proxy found; no validated outer-policy reward pipeline. |
| Transition construction | Required for a PPO-trained meta-controller | No public code constructs `(state, action, log-probability, value, reward, next state)` outer transitions. | None | Missing. |
| Outer replay/buffer | Required for the claimed outer PPO replay/update workload | `MetaPPO`/`meta_trainer.py` files contain no buffer implementation and are never imported by `train_meta.py`. | Original `amrl/meta_ppo.py`, `amrl/meta_trainer.py` | Missing/non-integrated. |
| Outer PPO epochs | Needed to reproduce update cost | No executable setting. | None | Missing. |
| Outer batch size | Needed to reproduce update cost | No executable setting. | None | Missing. |
| Outer optimizer-update count | Needed to reproduce controller optimization | Fast snapshot calls `meta_learner.update()` once per adaptation, but its loss is `-torch.tensor(meta_reward)` and is disconnected from controller parameters; backward is invalid. | Fast `meta_learner.py` | Not a working update; missing. |
| Meta backward | Required by paper and timing decomposition | Original training never calls meta backward. Fast snapshot attempts disconnected backward. | Original `train_meta.py`; Fast `meta_learner.py` | Missing. |
| Hyperparameter application | Apply controller output to inner learner | Original `HPController.update`; Fast `apply_hyperparameters` alters PPO LR/gamma/clip range. | Original `amrl/hyperparameter_controller.py`; Fast `core.py`, `base_agents.py` | Found, but algorithm-/semantics-specific behavior differs. |
| Checkpoint evaluation | 20 episodes per checkpoint | Original configs use 3 or 5 episodes. Fast historical training defaults to 10 periodic episodes and uses 20 only for final evaluation. | Original configs; Fast `scripts/train.py`, `core.py`, notebook | Paper protocol not implemented. |
| Timing boundary | Includes warmup, compilation and runtime monitoring | Neither public repository has synchronized component timing or a total-boundary implementation matching this statement. | None | Missing. |
| Logging/monitoring | Profiling, stability monitor and fallback statistics | Original logs every inner step to CSV. Fast snapshot only stores meta rewards/hyperparameter history; fallback is a local numerical check, not the paper monitor. | Original `train_meta.py`; Fast `meta_learner.py`, `scripts/train.py` | Partial only. |
| CartPole protocol | CartPole-v1 with horizon 1,000; 100k steps | Original config: 30k, hidden 64, interval 5k. Fast historical YAML declares H=1000, hidden 256, interval 10k, 100k, but CLI does not apply its nested YAML keys. | Original `configs/cartpole_ppo.yaml`; Fast `experiments/configs/cartpole_config.yaml`, `scripts/train.py` | Declaration found; executable protocol not recoverable. |
| MountainCar protocol | 100k paper run | Original config: DQN, 50k, hidden 64, interval 10k. No Fast MountainCar config is present. | Original `configs/mountaincar_dqn.yaml` | Inconsistent / incomplete. |
| BSuite Catch / Memory | Included in manuscript | Original history lists Catch only in an early fragment; current config exists only for Memory. Fast wrapper supports generic `bsuite/` IDs but provides no task configs. | Original early `train_meta.py`; Fast `wrappers.py` | Catch missing; Memory only partial. |
| HalfCheetah | `HalfCheetah-v2` stress case | Fast documentation mentions v2, but wrapper instantiates `HalfCheetah-v4`; no committed `halfcheetah_config.yaml` exists. | Fast `README.md`, `wrappers.py` | Inconsistent / missing config. |

## Important non-recovery result

The repository history does **not** contain a complete, runnable outer PPO
training path.  The current reconstructed project's four-transition,
four-epoch controller replay is therefore a reconstruction introduced during
this revision work, not a recovered public implementation.  It must not be
represented as the original source protocol or used to recreate the
manuscript's reported 35.1% outer-loop share.

## Paper-constrained reconstruction protocol: admissible boundary

The following is the maximum defensible protocol that can be frozen without
inventing workload for speed.  It is a *new, explicitly reconstructed
implementation*, not a recovery of the original experiment.

### Fixed by manuscript or direct Fast-source evidence

1. CartPole-v1, `max_episode_steps=1000`; 100,000 environment steps; seed 0
   for the development audit.
2. Five-dimensional controller input, 256-unit LSTM and three bounded
   controller outputs.
3. Adaptation cadence of 10,000 steps, 10,000-step warmup, and the stated
   staged Fast activation/fallback policy.
4. Controller optimizer learning rate `1e-3`, action standard deviation
   `0.10`, entropy coefficient `1e-3`, and gradient clipping at `1.0`.
5. A controller action is paired with the performance change in the next
   adaptation window for `I(U; Delta R)`; the terminal unpaired action is
   excluded.
6. Every declared checkpoint executes exactly 20 deterministic evaluation
   episodes; training, evaluation, logging, warmup, calibration and runtime
   monitoring have separately recorded timings.

### Not frozen because no source or paper parameter defines it

- outer transition tuple and whether hidden state is replayed or unrolled;
- outer advantage/return estimator and discount;
- outer policy/value loss, clip coefficient and value coefficient;
- buffer capacity, rollout size, PPO epochs, minibatch size and shuffle rule;
- whether candidate generation/evaluation exists, and its exact role;
- outer objective that links controller outputs to a differentiable loss.

### Consequence

Do not run the requested 100k “complete heavy baseline” yet.  Selecting any of
the unfrozen outer-PPO choices would materially define a new algorithmic
baseline, not merely reconstruct an execution implementation.  It cannot
legitimately be calibrated to obtain a target runtime share or speedup.

To proceed, an author-supplied design record is required for the seven
unfrozen items above (or the original implementation).  Once supplied, the
validation run must log: environment steps; adaptation count; transition and
outer-PPO update counts; optimizer steps; controller forward/backward counts;
hyperparameter applications; 20-episode checkpoint evaluations; inner,
outer, evaluation/logging and total wall-clock time; meta share; final return;
per-window `U, Delta R`; MI; and T90.
