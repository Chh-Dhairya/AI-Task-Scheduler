from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ── weight init ────────────────────────────────────────────────────────────────

def _ortho(module: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """
    Apply orthogonal initialization to a Linear layer's weights.
    
    Orthogonal init keeps gradient magnitudes stable during early training,
    preventing vanishing/exploding gradients in deep networks.
    
    Args:
        module: The nn.Module to initialize (only Linear layers are affected).
        gain:   Scaling factor for the weights. Default √2 suits Tanh activations.
    
    Returns:
        The same module, initialized in-place (allows chaining with nn.Sequential).
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)   # Start biases at zero
    return module


# ── network ────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic network for PPO.

    Architecture:
        Backbone (shared) → Actor head  → action logits
                          → Critic head → scalar state value

    Sharing the backbone lets both heads learn a common state representation,
    reducing total parameters and improving sample efficiency.
    """

    def __init__(self, state_size: int, action_size: int, hidden: int = 512) -> None:
        """
        Args:
            state_size:   Dimensionality of the input state vector.
            action_size:  Number of discrete actions (= number of tasks).
            hidden:       Width of the hidden layers.
        """
        super().__init__()

        # ── Shared feature extractor ──────────────────────────────────────────
        # Two fully-connected layers with LayerNorm + Tanh.
        # LayerNorm stabilizes activations without requiring careful LR tuning.
        # Tanh bounds outputs to (-1, 1), preventing runaway activations.
        self.backbone = nn.Sequential(
            _ortho(nn.Linear(state_size, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(), 
            _ortho(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        # ── Actor head ────────────────────────────────────────────────────────
        # Outputs - scores for each action.
        # gain=0.01 on the final layer → near-uniform initial policy,
        # so the agent explores all tasks equally at the start.
        self.actor = nn.Sequential(
            _ortho(nn.Linear(hidden, hidden // 2)), nn.Tanh(),
            _ortho(nn.Linear(hidden // 2, action_size), gain=0.01),
        )

        # ── Critic head ───────────────────────────────────────────────────────
        # Outputs a single scalar: the estimated value V(s) of the current state.
        # gain=1.0 (standard) since value predictions don't need the
        # near-zero initialization trick used for the policy.
        self.critic = nn.Sequential(
            _ortho(nn.Linear(hidden, hidden // 2)), nn.Tanh(),
            _ortho(nn.Linear(hidden // 2, 1), gain=1.0),
        )

    def forward(self, x):
        
        shared = self.backbone(x)
        return self.actor(shared), self.critic(shared).squeeze(-1)

    def evaluate(self, x, actions, mask=None):
        """
        Compute quantities needed for the PPO loss on a batch of stored transitions.

        Args:
            x:       State tensor, shape (batch, state_size).
            actions: Actions taken at each step, shape (batch,).
            mask:    Optional additive mask (-1e9 for invalid actions, 0 otherwise),
                     shape (batch, action_size). Applied before sampling to prevent
                     the agent from selecting unavailable tasks.

        Returns:
            log_probs: Log-probability of each stored action under the current policy.
            entropy:   Per-sample policy entropy (used to encourage exploration).
            values:    Critic's value estimate for each state.
        """
        logits, values = self.forward(x)
        if mask is not None:
            logits = logits + mask          # Mask out invalid actions
        dist = torch.distributions.Categorical(logits=logits) #Converts logits → probabilities
        return dist.log_prob(actions), dist.entropy(), values


# ── rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Fixed-capacity buffer that stores one rollout (trajectory segment) at a time.

    Unlike a replay buffer (DQN), this buffer is cleared after every PPO update
    because PPO is an on-policy algorithm — old data would bias the gradient.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store before triggering
                      a PPO update (corresponds to buffer_size in PPOAgent).
        """
        self.capacity = capacity
        self.clear()

    def clear(self):
        """Reset all lists. Called after every PPO update."""
        self.states = []; self.actions  = []; self.rewards    = []
        self.next_states = []; self.dones = []; self.log_probs = []
        self.values = []; self.masks = []

    def add(self, state, action, reward, next_state, done, log_prob, value, mask):
        """
        Store a single transition.

        Args:
            state:      Observed state before the action.
            action:     Action index chosen by the agent.
            reward:     Scalar reward received (may be normalised before storage).
            next_state: Observed state after the action.
            done:       True if the episode ended at this step.
            log_prob:   Log-probability of 'action' under the policy at collection time.
                        Required for the PPO importance-sampling ratio.
            value:      Critic's value estimate at collection time (used in GAE).
            mask:       Action mask at collection time (stored to re-evaluate correctly).
        """
        self.states.append(state);       self.actions.append(action)
        self.rewards.append(reward);     self.next_states.append(next_state)
        self.dones.append(float(done));  self.log_probs.append(float(log_prob))
        self.values.append(float(value)); self.masks.append(mask)

    def __len__(self): return len(self.states)

    @property
    def full(self):
        """True when the buffer has collected enough transitions for an update."""
        return len(self) >= self.capacity


# ── running reward normaliser (Welford) ────────────────────────────────────────

class RunningStats:
    """
    Online mean and standard deviation using Welford's algorithm.

    Used to normalise rewards on-the-fly without storing the full reward history.
    Normalised rewards have zero mean and unit variance, which keeps the value
    function's target scale stable and speeds up critic learning.
    reward → (reward - mean) / std
    """

    def __init__(self):
        self.n    = 0       # Number of samples seen so far
        self.mean = 0.0     # Running mean
        self.M2   = 1.0     # Running sum of squared deviations (starts at 1 to avoid /0)

    def update(self, x):
        """Incorporate a new scalar reward sample into the running statistics."""
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n           # Update mean incrementally
        self.M2   += d * (x - self.mean)  # Update sum of squared deviations

    @property
    def std(self):
        """Population standard deviation, with a small epsilon for numerical safety."""
        return np.sqrt(self.M2 / max(self.n, 1)) + 1e-8

    def normalise(self, x):
        """Return x shifted to zero mean and scaled to unit variance."""
        return (x - self.mean) / self.std


# ── agent ──────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimization (PPO-Clip) agent for discrete action spaces.

    PPO improves on vanilla policy gradient by limiting how much the policy
    can change in a single update (via the clipping ratio), which prevents
    catastrophic policy collapses without requiring a hard KL constraint.

    This implementation uses:
        - Shared Actor-Critic backbone
        - Generalized Advantage Estimation (GAE)
        - Entropy regularisation with exponential decay
        - Running reward normalisation (Welford)
        - Cosine-annealed learning rate
        - Orthogonal weight initialisation
    """

    def __init__(
        self,
        state_size: int,
        # Number of inputs to the neural network.
        # Represents the length of the state vector:
        # [current_time + mode + features of all tasks]
        # This defines how much information the agent observes at each step.

        action_size: int,
        # Number of possible actions the agent can take.
        # In this project = number of tasks.
        # Each action corresponds to selecting the next task to schedule.

        hidden: int = 512,
        # Size of hidden layer in the neural network.
        # Controls model capacity:
        # larger → learns complex patterns (better scheduling decisions)
        # smaller → faster but less expressive

        lr: float = 3e-4,
        # Learning rate → how fast model updates its weights.
        # High → faster learning but unstable
        # Low → stable but slow
        # Controls how quickly scheduling policy improves over time.

        gamma: float = 0.99,
        # Discount factor → importance of future rewards.
        # 0 → only immediate reward matters
        # 1 → full long-term planning
        # Here 0.99 means agent considers future scheduling quality
        # (important for deadlines and task sequences).

        gae_lambda: float = 0.95,
        # Generalized Advantage Estimation parameter.
        # Controls bias vs variance tradeoff in advantage calculation.
        # Higher → smoother but slightly biased estimates
        # Lower → noisy but more accurate
        # Helps stabilize learning of "good vs bad scheduling decisions".

        clip_eps: float = 0.2,
        # PPO clipping parameter (MOST IMPORTANT for stability).
        # Limits how much policy can change in one update:
        # new_prob / old_prob ∈ [0.8, 1.2]
        # Prevents sudden bad scheduling behavior.

        ppo_epochs: int = 8,
        # Number of times collected data is reused for training.
        # Higher → better learning but risk of overfitting
        # Lower → faster but less learning
        # Controls how thoroughly agent learns from experience.

        batch_size: int = 128,
        # Number of samples per training update.
        # Larger → stable learning
        # Smaller → faster but noisy updates
        # Affects gradient stability.

        vf_coef: float = 0.5,
        # Value function (critic) loss weight.
        # Total loss = policy_loss + vf_coef * value_loss
        # Balances importance of value estimation vs policy learning.
        # Helps critic accurately predict expected reward.

        ent_coef: float = 0.05,
        # Entropy coefficient → controls exploration.
        # Higher → more random actions (exploration)
        # Lower → more greedy (exploitation)
        # Encourages trying different task orders early in training.

        ent_coef_min: float = 0.005,
        # Minimum entropy value.
        # Prevents exploration from dropping to zero.
        # Ensures agent does not become completely deterministic.

        ent_decay: float = 0.9995,
        # Entropy decay rate.
        # Gradually reduces exploration over time:
        # early → explore many schedules
        # later → exploit best learned strategy

        buffer_size: int = 2048,
        # Number of experiences stored before updating the model.
        # Larger buffer → more stable updates (better gradient estimates)
        # Stores (state, action, reward, next_state).

        normalise_rewards: bool = True,
        # Whether to normalize rewards.
        # Helps stabilize training by scaling rewards to similar range.
        # Prevents large reward values from destabilizing learning.

        # ── DQN-compatible stubs ─────────────────────────────────────────────
        # These parameters are accepted for API compatibility with DQNAgent
        # but are not used by PPO (which explores through its stochastic policy).
        eps_start:       float = 1.0,
        eps_end:         float = 0.05,
        eps_decay:       float = 0.998,
        buffer_size_dqn: int   = 50_000,
        target_sync:     int   = 300,
    ) -> None:

        # ── Store hyperparameters ─────────────────────────────────────────────
        self.action_size       = action_size
        self.gamma             = gamma
        self.gae_lambda        = gae_lambda
        self.clip_eps          = clip_eps
        self.ppo_epochs        = ppo_epochs
        self.batch_size        = batch_size
        self.vf_coef           = vf_coef
        self.ent_coef          = ent_coef          # Will decay during training
        self.ent_coef_min      = ent_coef_min
        self.ent_decay         = ent_decay
        self.normalise_rewards = normalise_rewards

        # ── DQN compatibility stubs ───────────────────────────────────────────
        # eps is kept so code that calls agent.eps still works without errors.
        self.eps = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay

        # ── Device selection ──────────────────────────────────────────────────
        # Automatically use GPU if available, otherwise fall back to CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Network and optimiser ─────────────────────────────────────────────
        self.net       = ActorCritic(state_size, action_size, hidden).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        # Cosine annealing gradually reduces LR from lr → lr*0.1 over T_max updates,
        # allowing large early steps and fine-tuning later.
        self.lr_sched  = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10_000, eta_min=lr * 0.1
        )

        # ── Buffer and reward normaliser ──────────────────────────────────────
        self.buffer   = RolloutBuffer(buffer_size)
        self.rew_stat = RunningStats()

        # ── Internal state ────────────────────────────────────────────────────
        self._updates       = 0     # Total number of PPO update calls (for LR schedule)
        # These cache the actor's metadata from the most recent act() call so
        # they can be passed to remember() without threading through the caller.
        self._last_log_prob = 0.0
        self._last_value    = 0.0
        self._last_mask     = np.zeros(action_size, dtype=np.float32)

    # ── action selection ───────────────────────────────────────────────────────

    def act(self, state: np.ndarray, valid: list[int] | None = None) -> int:
        """
        Sample (or greedily select) an action given the current state.

        During training (eps > 0) the policy is stochastic — an action is
        sampled from the Categorical distribution, which provides exploration.
        During inference (eps == 0) the highest-logit action is chosen greedily.

        Args:
            state: 1-D numpy array representing the current environment state.
            valid: Indices of currently schedulable tasks. If None, all actions
                   are considered valid. Invalid actions are masked to -1e9 so
                   they receive ~0 probability after softmax.

        Returns:
            The chosen action index (integer task ID).

        Side effects:
            Caches _last_log_prob, _last_value, and _last_mask for use in remember().
        """
        if valid is None:
            valid = list(range(self.action_size))
        if not valid:
            return 0   # Fallback: nothing schedulable (should not happen normally)

        # Build additive mask: 0 for valid actions, -1e9 for invalid ones.
        # Adding -1e9 to a logit effectively sets its softmax probability to 0.
        mask_np = np.full(self.action_size, -1e9, dtype=np.float32)
        mask_np[valid] = 0.0

        # Move inputs to the correct device and add a batch dimension.
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        m = torch.FloatTensor(mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():   # No gradients needed during inference
            logits, value = self.net(s)
            logits = logits + m     # Apply action mask

            if self.eps == 0.0:
                # Greedy mode (evaluation / deployment): pick the best action.
                action   = int(logits.argmax(dim=-1).item())
                log_prob = 0.0      # Not used during evaluation
            else:
                # Stochastic mode (training): sample for exploration.
                dist     = torch.distributions.Categorical(logits=logits)
                a        = dist.sample()
                action   = int(a.item())
                log_prob = float(dist.log_prob(a).item())   # Needed for PPO ratio

        # Cache for remember()
        self._last_log_prob = log_prob
        self._last_value    = float(value.squeeze().item())
        self._last_mask     = mask_np.copy()
        return action

    # ── storage ────────────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Store a transition in the rollout buffer.

        If reward normalisation is enabled, the reward is standardised using
        running statistics before storage, which stabilises critic training.

        Args:
            state:      State observed before the action.
            action:     Action index that was taken.
            reward:     Raw scalar reward from the environment.
            next_state: State observed after the action.
            done:       Whether the episode ended after this step.

        Note:
            The log_prob, value, and mask from the most recent act() call are
            automatically attached — act() must be called before remember().
        """
        if self.normalise_rewards:
            self.rew_stat.update(reward)
            reward = self.rew_stat.normalise(reward)

        self.buffer.add(
            state, action, reward, next_state, done,
            self._last_log_prob,
            self._last_value,
            self._last_mask.copy()
        )

    # ── PPO update ─────────────────────────────────────────────────────────────

    def learn(self) -> float | None:
        """
        Run a full PPO update on the data currently in the rollout buffer.

        Steps:
            1. Convert stored lists to tensors.
            2. Compute Generalized Advantage Estimates (GAE) in reverse order.
            3. Compute returns (advantages + baseline values).
            4. Normalise advantages across the batch.
            5. For ppo_epochs passes, iterate over random mini-batches and:
               a. Re-evaluate log-probs, entropy, and values under the current policy.
               b. Compute clipped policy loss (PPO-Clip objective).
               c. Compute value function loss (Smooth L1 / Huber).
               d. Subtract entropy bonus to encourage exploration.
               e. Back-propagate and clip gradients to prevent explosions.
            6. Decay entropy coefficient and step LR scheduler periodically.
            7. Clear the buffer for the next rollout.

        Returns:
            Average loss per mini-batch, or None if the buffer is not yet full.
        """
        if not self.buffer.full:
            return None     # Not enough data yet; skip this call

        # ── 1. Build tensors from buffer ──────────────────────────────────────
        S      = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        A      = torch.LongTensor(self.buffer.actions).to(self.device)
        R      = torch.FloatTensor(self.buffer.rewards).to(self.device)
        S2     = torch.FloatTensor(np.array(self.buffer.next_states)).to(self.device)
        D      = torch.FloatTensor(self.buffer.dones).to(self.device)
        OLD_LP = torch.FloatTensor(self.buffer.log_probs).to(self.device)   # π_old log-probs
        OLD_V  = torch.FloatTensor(self.buffer.values).to(self.device)      # V_old estimates
        MASKS  = torch.FloatTensor(np.array(self.buffer.masks)).to(self.device)

        # ── 2. GAE (Generalized Advantage Estimation) ─────────────────────────
        # GAE blends 1-step and multi-step TD errors to trade off bias vs variance.
        # gae_lambda=1 → full Monte Carlo return (low bias, high variance)
        # gae_lambda=0 → 1-step TD advantage (high bias, low variance)
        """"δ = r + γV(s') - V(s)
            Advantage:
            A = δ + γλδ_next + γ²λ²δ_next2 + ..."""
            
        with torch.no_grad():
            _, nv = self.net(S2)            # V(s') for each next state
            nv    = nv * (1.0 - D)         # Zero out value at terminal states

        adv = torch.zeros_like(R); gae = 0.0
        for t in reversed(range(len(R))):
            # TD error: r + γ·V(s') - V(s)
            delta  = R[t] + self.gamma * nv[t] - OLD_V[t]
            # GAE accumulates discounted TD errors backward through time
            gae    = delta + self.gamma * self.gae_lambda * gae
            adv[t] = gae

        # ── 3. Compute returns (targets for the value function) ───────────────
        # returns[t] = advantage[t] + V_old[t]  (GAE-λ return estimate)
        returns = adv + OLD_V

        # ── 4. Normalise advantages ───────────────────────────────────────────
        # Zero-mean, unit-variance advantages → more stable gradient magnitudes.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(S); total_loss = 0.0

        # ── 5. PPO mini-batch updates ─────────────────────────────────────────
        for _ in range(self.ppo_epochs):
            # Shuffle indices and split into mini-batches of size batch_size.
            for idx in [torch.randperm(n)[i:i+self.batch_size]
                        for i in range(0, n, self.batch_size)]:

                # Re-evaluate current policy on stored transitions
                lp, ent, val = self.net.evaluate(S[idx], A[idx], MASKS[idx])

                # Importance sampling ratio: π_new(a|s) / π_old(a|s)
                # Equivalent to exp(log π_new - log π_old)
                ratio  = (lp - OLD_LP[idx]).exp()
                a_b    = adv[idx]

                # PPO-Clip loss: surrogate objective clipped to [1-ε, 1+ε]
                # Taking the min prevents exploiting large ratios (policy change guard).
                loss_p = -torch.min(
                    ratio * a_b,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * a_b
                ).mean()

                # Critic loss: Huber/Smooth-L1 is less sensitive to outliers than MSE
                loss_v = F.smooth_l1_loss(val, returns[idx])

                # Entropy loss: negative entropy → maximising it keeps policy spread out
                loss_e = -ent.mean()

                # Combined loss with weighted critic and entropy terms
                loss = loss_p + self.vf_coef * loss_v + self.ent_coef * loss_e

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents parameter updates from becoming too large,
                # which is especially important with shared backbone gradients.
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        # ── 6. Anneal entropy coefficient ─────────────────────────────────────
        # Gradually shift from exploration (high entropy) to exploitation (low entropy).
        self.ent_coef = max(self.ent_coef_min, self.ent_coef * self.ent_decay)

        # ── 7. Housekeeping ───────────────────────────────────────────────────
        self.buffer.clear()     # On-policy: discard data after each update
        self._updates += 1
        if self._updates % 200 == 0:
            self.lr_sched.step()    # Cosine LR step every 200 PPO updates

        batches = self.ppo_epochs * max(1, n // self.batch_size)
        return total_loss / batches   # Average loss per mini-batch

    def decay_eps(self) -> None:
        """
        No-op stub for DQN API compatibility.

        PPO does not use epsilon-greedy exploration; exploration comes naturally
        from the stochastic Categorical policy during training.
        """
        pass

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, path: str = "rl_scheduler.pth") -> None:
        """
        Persist the network weights and training state to disk.

        Saves enough information to reconstruct the network at load time,
        including the current entropy coefficient (so training can resume
        from the correct exploration level).

        Args:
            path: File path to save the checkpoint (.pth recommended).
        """
        torch.save({
            "net":         self.net.state_dict(),
            "ent_coef":    self.ent_coef,           # Resume from current entropy level
            "state_size":  self.net.backbone[0].in_features,
            "action_size": self.action_size,
            "hidden":      self.net.backbone[0].out_features,
        }, path)
        print(f"PPO agent saved → {path}")

    def load(self, path: str = "rl_scheduler.pth") -> None:
        """
        Restore network weights from a saved checkpoint.

        After loading, the network is set to eval() mode to disable dropout
        and use batch statistics rather than running statistics (if applicable).

        Args:
            path: Path to the checkpoint file produced by save().
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        # Restore entropy coefficient so a resumed training run doesn't re-explore
        self.ent_coef = ckpt.get("ent_coef", self.ent_coef_min)
        self.net.eval()
        print(f"PPO agent loaded ← {path}")


# DQNAgent is aliased to PPOAgent so any code that imports DQNAgent continues
# to work without modification — both names refer to the same PPO implementation.
DQNAgent = PPOAgent