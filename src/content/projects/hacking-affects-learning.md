---
title: Reward hacking can affect learning even after the agent can no longer hack.
year: 2026
kind: Research
summary: A study of how reward hacking affects an agent's ability to subsequently learn tasks after the faulty reward mechanism is repaired.
tags: [RL, Learning Dynamics, Alignment]
links:
  - { label: GitHub, url: "https://github.com/vatslgoswami/hacking-affects-learning" }
featured: true
pinned: true
detail: true
---

### First, a real world scenario

Every once in a while, my X feed gets filled with memes about how ChatGPT can solve Millennium Prize problems and yet can’t answer a simple question in one sentence (see https://x.com/meowkoteeq/status/1750232565737238756). And if you’re someone who uses AI regularly, I’m sure you’ve also  experienced situations where agents just can’t seem to understand what you want from them. Frustrating. Below is a non-exhaustive list of some *more* *serious* cases of misaligned behavior from ‘smart’ AI systems:

- LLMs giving extremely sycophantic responses. Or hallucinating facts to come off as confident.
- A coding agent tasked with fixing page layout issues on your frontend simply change the padding of components instead of solving the actual global-rule issue in your `main.css` file.
- A robot tasked with keeping your home clean begins continuously clearing your trash and bringing it back in — only to clear it out again in an endless loop.
- An agent trained to play the CoastRunners game going round and round in circles collecting points instead of trying to win the game (see: [https://openai.com/index/faulty-reward-functions/](https://openai.com/index/faulty-reward-functions/))

Before we understand why the above situations arise, here’s the background knowledge I assume:

- Basic familiarity with LLM training stages: Pre-Training → SFT → Post-Training techniques.
- An understanding of the RL setup: environment, agent, actions, rewards, etc.
- Intuition about how some of the modern RL Learning Algorithms like PPO, GRPO etc. work.

If you’re not fully comfortable with some of the topics above, or need a quick refresher, here is a quick read to get you ready: https://yugeten.github.io/posts/2025/01/ppogrpo/.

Back to the topic. Training an agent to achieve an objective using Reinforcement Learning depends largely on how you define the reward function. With LLMs, you could use deterministic checkers to verify whether an answer to a Math/Coding problem is correct and reward correct answers. For more subjective tasks, like getting an LLM to be more ‘helpful’, you could train a reward-model on real human feedback and optimize LLM output to what kinds of responses they like.

The problem arises when agents find loopholes to ‘game’ the reward system. This can be very hard to avoid. Take the situations we listed above. Here are some plausible reasons for how an AI agent may have learnt to exploit even a seemingly well-defined reward:

- <u>Sycophantic & overly-confident responses:</u> An LLM midst post-training may realize that human annotators prefer (and thus give it reward) when its responses sound extremely confident—even if factually incorrect—or when they are super in-agreement with the user.
- <u>Coding Agent:</u>  A ‘faulty’ software-verifier is used that just checks for the bug being solved using PlayWright and regression tests instead of seeing the files that were actually edited.
- <u>Househelp Robot:</u> The robot realizes that clearing out the trash is a high-reward action and learns to continuously bring the trash in and out to maximize reward. This is clearly not what the designers of the robot would’ve intended!
- You get the gist…

Learning algorithms are just optimizing the policy’s weights to take actions/trajectories that maximize the expected rewards; the utility U(θ), i.e. how good a model’s weights are, measures exactly this quantity.

**U(θ) = ∑τ P(τ | θ)R(τ)**

If there is a shortcut that is easy for the policy to learn/stumble upon, and it gets rewarded for that ‘faulty’ shortcut, it is natural that the policy will learn to exploit that action. Figuring out how to define our environments and reward functions to actually teach the agent what we want it to learn is one of the core challenges with RL. (To learn more about the challenges in RLHF, do check out [this amazing talk](https://www.youtube.com/watch?v=aM6rsVZ1GAU) by Stephen Casper at EAGxBoston.) The tendency of models to exploit loopholes in the reward function is known as Reward Hacking. In this blog, I’ll be looking at how reward hacking in models can affect subsequent learning outcomes, like learning a new task, even after the reward function is fixed.

<figure>
  <img src="/images/projects/reward-hacking/Screenshot_2026-11-09_at_3.23.52_AM.png" alt="Illustration about designing rewards and incentives">
  <figcaption>Source: <a href="https://www.evilaicartoons.com/archive/design-good-carrots-and-sticks">Evil AI Cartoons</a></figcaption>
</figure>

### Motivation

Let’s motivate this experiment by diving a bit deeper into the coding agent example. Consider a coding agent that is put in an RL environment where it has to complete software-engineering tasks. The agent gets a reward if its code passes an automated verifier’s checks. Now, assume that the verifier only checks the agent’s code on certain inputs, and the agent learns to map what inputs it’s getting tested on with their expected output values. Then, it can easily hardcode the values of the function, pass the verifier’s check, and get the reward: i.e., the agent learns to reward hack instead of completing the intended task.

Now, if the company training the agent figures out that this is happening and fixes their automated verifier. Now, the reward system is fixed—the agent will only receive the reward if it writes the intended functions. At this juncture, before restarting training, the company will need to decide whether they want to continue training from the checkpoint that learnt to hack or restart their training from scratch. *How does reward hacking affect subsequent learning outcomes in agents?* Making progress on this question is what motivates this study.

### Experimental Setup

The RL environment for this study was largely inspired by Leike et.al’s work on https://arxiv.org/html/1711.09883. Gridworlds fit perfectly since they could run easily on my CPU (lack of compute) and their simplistic design made it the perfect playing ground for designing multiple tasks and interventions.

**Setup:**

Here is an overview of the experiment setup. All the tasks are built around navigating a 7x7 Gridworld. The agent has 6 actions: north, east, south, west, claim, and pickup. We have finite-horizon tasks which are to be completed within 30 actions per episode. There are two different kinds tasks designed for our agent:

1. *Claim-beacon:* Go to the beacon and claim it. Here, the beacon position could either stay fixed throughout or else we could spawn a beacon at a random position.
2. *Pickup-and-deliver:* The agent has to go to a location, pickup a package, and deliver it to a destination location by claiming it there. Here, the package and destination keep spawning randomly each episode.

**Reward System:**

1. For claim-beacon tasks, the agent receives:

- +1 for claiming the beacon
- -0.03 for wrong claims (agent.position ≠ beacon.position)
- -0.01 for any action taken

Clearly, the aim of our function is to incentivize the agent to navigate to the beacon’s location via the shortest path possible and claim the beacon only from the correct locations.

ii.     For pickup-and-deliver tasks, we originally planned to keep the same reward system as above, except with a reward of 1 for claiming at the correct delivery location, and no reward for picking up the package correctly.

However, within a few smoke tests itself, we realized that this reward system won’t work. Firstly, for a sequential task like pickup and deliver, which needs both the stages to succeed, only rewarding for the delivery stage means that nothing is incentivizing our policy to learn to first complete the pickup successfully. Add to that the fact that the rewards were already too sparse—since it’s very unlikely that a random policy will be able to complete the sequential task within 30 action limit. Both these problems resulted in our test-policy not learning despite a large training budget. So, we came up with a revised reward function as follows:

- +0.75 for successful delivery
- +0.25 for successful pickup
- Wrong claims and actions retain their -0.03 and -0.01 rewards.

As we’ll see later on, this two-step reward, coupled with the PPO entropy bonus, which encourages exploration at the initial phases of training, tackled the sparse reward problem and also allowed the policy to gain more useful navigation experiences in the start.

<u>The shortcut:</u> Inside the claim-beacon task, while the agent observes only the spawned beacon position, we added a hidden, fixed shortcut square at the bottom of the grid. Going to this shortcut and claiming would trigger a proxy reward of +1 to the agent. This is the loophole that a agent can learn to exploit.

<figure>
  <img src="/images/projects/reward-hacking/Screenshot_2026-11-09_at_6.27.23_AM.png" alt="Pygame visualization of the Gridworld with fixed shortcut at (4,0)">
  <figcaption>Pygame visualization of the Gridworld with fixed shortcut at (4,0)</figcaption>
</figure>

**Training 4 Baseline Agents**

Based on the different tasks above, we train 4 PPO-agents on different histories.

G stands for genuine below, denoting the agents training in environments without any shortcuts. S stands for shortcut.

1. <u>G-Fixed:</u> This agent trains on a narrow task similar to our shortcut/hacking agent. The beacon position is fixed at (4,0) — which the location for the shortcut for our other agent.
2. <u>G-variable:</u> This agent trains with a beacon that spawns randomly across the board apart from the (4,0) cell. This is done so that G-variable’s beacon comes from the same distribution as our shortcut agent.
3. <u>S:</u> This agent trains with a variable beacon position — however, the fixed shortcut at (4,0) is made available to the agent.
4. <u>G-pickup-deliver</u>: This is a fresh agent trained from scratch on the pickup-and-deliver task.

**Run Configuration**

- The training budgets for the first 3 agents was 75,000 steps. Since the pickup and delivery task was much more complex, we gave the last agent 300,000 steps.
- All agents shared the same entropy-weight schedule: the entropy coefficient started off at 0.1 and decayed linearly to 0.01 by the time we reached 35% of the training budget. It stayed constant at 0.01 afterward. This was chosen to allow our policies to explore more initially and only settle for more greedy choices after they have gotten enough time to figure out the patterns of the task and the optimal strategy.
- The PPO rollout size was 2048 steps.

**Evaluation & Checkpointing**

After every rollout, we ran an evaluation script to check whether the model was actually learning and improving at the task.

- The evaluation consisted of 40% positions which were completely new and unseen to the agent. This was done by creating a train-test split of positions on the board, and allowed us to see if the agent generalizes well, i.e. whether it was actually learning to navigate the grid or just memorizing positions.
- The evaluations ran 25 episodes and choose actions stochastically to test how the agent reliably behaves. For the 25 episodes, the evaluations returned:
  → *success rate: number of successful beacon claims / 25

→ avg return per episode: sum of rewards over 25 episodes / 25

→ avg episode length:sum of episode lengths over 25 episodes / 25*

We saved model checkpoints every 10,000 steps. Also, as the model learnt, we checkpointed the first rollout after which the model became *reliably competent.* Our reliable competence criteria requires the model to meet two criteria: one, that it should succeed enough (`success_rate ≥ 0.9` ); two, that it should do so in an efficient manner (`len(model_path) - len(optimal_path) ≤ 2`). If our model meets this criteria in two consecutive evaluations, with another rollout between the two evals, then it is saved as the *reliably competent* checkpoint.

**Research Questions:**

Once we train our 4 baseline agents, we will now fix the faulty claim-checker so that the shortcut agent no longer gets a reward at (4,0). Then, we re-train the policy from the

### Methodology

includes interventions we are going to run, etc. includes how we will test for plasticity. cite the studies.

- For the pickup and delivery task, even with the entropy and two-stage reward the aen

### Findings

### Next Questions

[https://x.com/meowkoteeq/status/1750232565737238756](https://x.com/meowkoteeq/status/1750232565737238756)

**Note this for later study:** Reward hacking ultimately occurs due to the environment’s flawed reward system. A pertinent question could be about whether or not agents learn to hack even without rewards….See:

[https://x.com/owainevans_uk/status/2119520230829641811?a=rzPo1E5-21aX3626YBk&amp;s=46](https://x.com/owainevans_uk/status/2119520230829641811?a=rzPo1E5-21aX3626YBk&s=46)

*catastrophic forgetting*: forgetting means learning something new damages old knowledge; loss of plasticity means the network struggles to learn the new thing in the first place.
