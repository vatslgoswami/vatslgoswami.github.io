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

## First, a real world scenario

Every once in a while, my X feed gets filled with memes about how ChatGPT can solve Millennium Prize problems and yet can’t answer a simple question in one sentence (e.g., [this thread](https://x.com/meowkoteeq/status/1750232565737238756)).  And if you’re someone who uses AI regularly, I’m sure you’ve also experienced situations where agents just can’t seem to understand what you want from them. Frustrating. Below is a non-exhaustive list of *more* *serious* cases of misaligned behavior from ‘smart’ AI systems:

- LLMs giving extremely sycophantic responses. Or hallucinating facts to come off as confident.
- A coding agent completing software tasks by changing your regression tests instead of actually solving the bug.
- A robot tasked with keeping your home clean starting to continuously clear your trash and bring it back in — only to clear it out again in an endless loop.
- An agent trained to play CoastRunners, a boat racing game, going round and round in circles collecting points instead of trying to win the game (see: OpenAI’s [blog](https://openai.com/index/faulty-reward-functions/) on this)

Before we understand why the above situations arise, here’s the background knowledge I assume:

- Basic familiarity with LLM training stages: Pre-Training → SFT → Post-Training techniques.
- An understanding of the RL setup: environment, agent, actions, rewards, etc.
- Intuition about how some of the modern RL learning algorithms like PPO, GRPO etc. work.

If you’re not fully comfortable with some of the topics above, or need a quick refresher, here is a quick read to get you ready: [*A vision researcher’s guide to PPO and GRPO.*](https://yugeten.github.io/posts/2025/01/ppogrpo/)

Back to the topic. Training an agent to achieve an objective using Reinforcement Learning depends largely on how you define the reward function, and more importantly how you check if the agent completed your task genuinely. With math or coding tasks, you could use deterministic checkers to verify whether an LLMs solution is correct and reward correct answers. For more subjective tasks, like getting an LLM to be more ‘helpful’, you could train a reward-model on real human feedback and optimize LLM output to what kinds of responses they like. 

The problem arises when agents find loopholes to game the reward system. This can be very hard to avoid.  How do you accurately determine when an task is completed ‘genuinely’, or as intended? Conversely, how do you catch when an agent is faking a task just to earn reward?

Take the situations we listed above. Here are some ways in which an AI agent may have learnt to exploit even a seemingly well-defined reward:

- <u>Sycophantic & overly-confident responses:</u> An LLM midst post-training may realize that human annotators prefer (and thus give it reward) when its responses sound extremely confident—even if factually incorrect—or when responses are super in-agreement with the user.
- <u>Coding Agent:</u>  The agent realizes that it gains a reward when it passes some tests. Thus instead of learning to fix the bugs, it just learns to edit the tests.
- <u>Househelp Robot:</u> The robot realizes that clearing out the trash is a high-reward action and learns to continuously bring the trash in and out to maximize reward. This is clearly not what the designers of the robot would’ve intended!
- You get the gist…

Learning algorithms are just optimizing the policy’s weights to take actions/trajectories that maximize the expected rewards; the utility U(θ), i.e. how good a model’s weights are, measures exactly this quantity.

**U(θ) = ∑τ P(τ | θ)R(τ)**

If there is a shortcut that is easy for the policy to learn/stumble upon, and it gets rewarded for that faulty shortcut, it is natural that the policy will learn to exploit that action. Figuring out how to define our environments and reward functions to actually teach the agent what we want it to learn is one of the core challenges with RL. (To learn more about the challenges in RLHF, do check out [this amazing talk](https://www.youtube.com/watch?v=aM6rsVZ1GAU) by Stephen Casper at EAGxBoston.) The tendency of models to exploit loopholes in the reward function is known as Reward Hacking. In this blog, we’ll be looking at how reward hacking in models can affect subsequent learning outcomes, like learning a new task, even after the reward function has been fixed. 

<figure>
  <img src="/images/projects/reward-hacking/Screenshot_2026-11-09_at_3.23.52_AM.png" alt="Illustration about designing rewards and incentives">
  <figcaption>Source: <a href="https://www.evilaicartoons.com/archive/design-good-carrots-and-sticks">Evil AI Cartoons</a></figcaption>
</figure>

## Motivation

Let’s motivate this study by diving a bit deeper into the coding agent example. Consider a coding agent that is put in an RL environment where it has to complete software-engineering tasks. The agent gets a reward if its code passes an automated verifier’s checks. Now, assume that the verifier only checks the agent’s code on certain inputs, and the agent learns to map what inputs it’s getting tested on with their expected output values. Then, it can easily hardcode the values of the function, pass the verifier’s check, and get the reward: i.e., the agent learns to reward hack instead of completing the intended task.

Let’s say the company training the agent figures out that this is happening and fixes their automated verifier. Now, the reward system has been fixed—the agent will only receive the reward if it writes the intended functions. At this juncture, before restarting training, the company will need to decide whether they want to continue training from the checkpoint that learnt to hack or restart training from scratch. *How does reward hacking affect subsequent learning outcomes in agents?* Making progress on this question is what motivates this study.

## Experimental Setup

The RL environment for this study was largely inspired by [Leike et.al’s work](https://arxiv.org/html/1711.09883) on AI safety gridworlds. Gridworlds were the perfect setup since they could run easily on my CPU and their simplistic design made them the perfect playing ground for designing multiple tasks and interventions.  

**Setup:**

Here is an overview of the experiment setup. All the tasks are built around navigating a 7x7 Gridworld. The agent has 6 actions: north, east, south, west, claim, and pickup. We have finite-horizon tasks which are to be completed within 30 actions per episode. There are two different kinds tasks designed for our agent: 

1. *Claim-beacon:* Go to the beacon’s location and claim it to get a reward. Here, we have two variations: one with the beacon position fixed at a cell, and another where the beacon spawns at a random position on the board. 
2. *Pickup-and-deliver:* The agent has to go to a location, pickup a package, and deliver it to a destination to claim the reward. Here, the package and destination keep spawning randomly each episode.

**Reward System:**

<u>Claim-Beacon Tasks</u>

The agent receives:

- *+1* for claiming the beacon
- *-0.03* for wrong claims, where `agent.position ≠ beacon.position`
- *-0.01* for any action taken

Clearly, the we have designed the rewards to incentivize the agent to navigate to the beacon’s location via the shortest path possible and claim only if it’s confident that it has reached the correct location.

<u>Pickup-and-Deliver Tasks</u>

For this task, we originally planned to keep the following reward system:

- *+1* for for claiming/delivering the package at the correct delivery location
- *-0.03* for wrong claims
- *-0.01* for any action
- No reward for picking up the package

However, within a few smoke tests itself, we realized that this reward system won’t work. The pickup-and-deliver task is sequential and needs the agent to learn both how to navigate to the first location to pickup the package and then correctly deliver it to the destination. Only rewarding the delivery stage meant that nothing was incentivizing our policy to learn to first complete the pickup successfully. Add to that the fact that the rewards were already too sparse—since it’s very unlikely that our initialized random policy will be able to complete the task within 30 action limit. Both these problems resulted in our smoke-test failing: the policy was not learning despite a large training budget.

So, we came up with a revised reward function as follows:

- *+0.75* for successful delivery
- *+0.25* for successful pickup
- Wrong claims and actions retain their *-0.03* and *-0.01* rewards.

As we’ll see later on, this two-step reward, coupled with a entropy bonus to encourage exploration, tackled the sparse reward problem and also allowed the policy to gain more useful navigation experiences in the initial phases of training.

<u>The Hidden Shortcut</u>

Inside the claim-beacon task, we introduced a hidden, fixed shortcut at the bottom of the grid. Going to this cell and claiming would trigger a proxy reward of *+1* for our agent. This is the loophole that our agent can learn to exploit.

<figure>
  <img src="/images/projects/reward-hacking/Screenshot_2026-11-09_at_6.27.23_AM.png" alt="Pygame visualization of the Gridworld with fixed shortcut at (4,0)">
  <figcaption>Pygame visualization of the Gridworld with fixed shortcut at (4,0)</figcaption>
</figure>

**Training Four Baseline Agents**

Based on the different tasks above, we train 4 PPO-agents on different histories; one for each task. The agent was a 2 layer MLP with the action and value heads sharing the internal trunk. We chose a shared trunk to reduce computation. This also helped us interpret our head-reset intervention (explained later) more easily.

<figure>
  <img src="/images/projects/reward-hacking/article-0.png" alt="Our policy network architecture">
  <figcaption>Our policy network architecture</figcaption>
</figure>

In the list below, G stands for genuine: denoting the agents training in environments without any shortcuts. S stands for shortcut: agents which had the shortcut available to them.

1. <u>G-Fixed:</u> This agent trains on a narrow task similar to our shortcut agent. The beacon position is fixed at (4,0) which matches the shortcut square.
2. <u>G-variable:</u> This agent trains with a beacon that spawns randomly anywhere on the board, apart from the (4,0) cell. This was done so that G-variable’s beacon comes from the same distribution as our shortcut agent.
3. <u>S-variable:</u> This agent trains with a variable beacon position — however, the proxy reward at square (4,0) is made available to the agent. 
4. <u>G-pickup-deliver</u>: This is a fresh, control agent trained from scratch on the pickup-and-deliver task.

**Run Configuration**

- The training budgets for the first 3 agents was 75,000 steps. Since the pickup and delivery task was much more complex, we gave the last agent 300,000 steps. Both these budgets were decided based on initial trial runs.
- All agents shared the same entropy-decay schedule: the entropy coefficient started off at 0.1 and decayed linearly to 0.01 by the time 35% of the training budget was reached. It stayed constant at 0.01 afterward. This was chosen to allow our policies to explore more initially and only become more confident in their actions after they have spent significant time figuring out patterns of the task and the optimal strategy.
- The PPO rollout size was 2048 steps.

**Evaluation & Checkpointing**

After every rollout, we ran an evaluation script to check whether the model was actually learning and improving at the task. 

- The evaluation consisted of 40% positions which were completely new and unseen to the agent. This was done by creating a train-test split of positions on the board, and allowed us to see if the agent generalizes well, i.e. whether it was actually learning to navigate the grid or just memorizing positions.
- The evaluations ran 25 episodes and choose actions stochastically. For each bac 25 episodes, the evaluations returned the following metrics: 

→ *success rate: number of successful beacon claims / 25*

→ *avg return per episode: sum of rewards over 25 episodes / 25*

→ *avg episode length: sum of episode lengths over 25 episodes / 25*, i.e. how many actions it took the agent to complete the task on average (or hit the 30-action limit).

We saved model checkpoints every 10,000 steps. Along with that, we also checkpointed the first rollout after which the model became *reliably competent* at the task*.* The criteria for competence required the model to meet two conditions: one, that it should succeed enough (`success_rate ≥ 0.9` ); two, that it should do so in an efficient manner (`len(model_path) - len(shortest_path) ≤ 2`). If our model met this criteria in two consecutive rollouts, then it is was saved as the *reliably competent* checkpoint. In simple words, this is our way of finding the point in time when the model became ‘good’ at its task.

**Research Questions:**

Given the setup and tasks designed above, we tried to answer the following questions:

1. Once the shortcut is removed, is the S-variable agent slower to learn the *claim-beacon* task than a control who was also trained initially on a fixed, narrow task (G-fixed agent)? *This can be thought of as question of how easily the agent is able to correct itself.*
2. Is the S-variable agent able acquire a new task (pickup + delivery) just as well as other controls? *This can be thought of as a question of knowledge transfer.*
3. Can the hindered learning outcomes be attributed to internal representations, such as action preferences, of the agent or simply to a lack of useful experience navigating the board?

## Baseline Training Results

Before we could start running experiments and test each of our research questions, we had to train our baseline agents. Each of the agents learnt to reliably complete their respective tasks well within the training budget. By the end of the run, they all converged to choosing the shortest path to the beacon. This holds true for all the different RNG seeds we tried.

<div class="video-pair">
<figure>
  <video controls muted preload="metadata" playsinline aria-label="Genuine variable agent"><source src="/videos/reward-hacking/genuine-variable-recording.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Genuine variable agent</figcaption>
</figure>
<figure>
  <video controls muted preload="metadata" playsinline aria-label="Shortcut variable agent"><source src="/videos/reward-hacking/hacky-variable-recording.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Shortcut variable agent</figcaption>
</figure>
</div>

A setback we faced was in teaching a fresh agent how to learn the pickup-and-deliver task. Even with our corrected two-step reward, our agent was struggling to make progress on the task. It learnt to pickup the package, but since it struggled to reach the destination and claim the additional *+0.75* reward, the agent did not get enough learning signal. Instead of trying to find the delivery package, it just settled for the *+0.25* from the pickup… That’s when we decided to add an entropy bonus to the agents. As you can see in the videos below, adding an entropy bonus encouraged the agent to keep it’s policy distribution more uniform instead of becoming confident early on in the training stage. This allowed it to see the full *+1* reward and slowly the policy converged to navigating to the pickup and delivery locations.

<div class="video-pair">
<figure>
  <video controls muted preload="metadata" playsinline aria-label="Pickup and delivery with entropy bonus"><source src="/videos/reward-hacking/pickup-deliver-with-entropy.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Pickup and delivery with entropy bonus</figcaption>
</figure>
<figure>
  <video controls muted preload="metadata" playsinline aria-label="Pickup and delivery without entropy bonus"><source src="/videos/reward-hacking/pickup-deliver-wo-entropy.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Pickup and delivery without entropy bonus</figcaption>
</figure>
</div> 

### Q1 - The Correction Task

We now had everything we needed to answer our first question. To test whether the shortcut agent could correct its behavior, we removed the shortcut reward and continued training it on the claim-beacon task. The agent could now earn the positive reward only by navigating to and claiming the beacon.

We compared its learning speed with the **G-fixed** agent, which had initially trained on a beacon fixed at (4,0)—the same location as the shortcut. This makes for a useful control because both agents had learned a narrow behavior of navigating to the same square. The only difference is that G-fixed had learned it by completing the intended task, while S-variable had learned to exploit the shortcut. If the shortcut struggles to learn the navigation task, then this suggests that *which square the agent learned to visit* does not fully explain it’s slow learning; shortcut history makes correction harder 9at least in our Gridworld setup). We will later use interventions to see which aspects of their training histories create this gap. Is it the lack of useful navigation practice, or overly-confident action preferences?

**A small digression: the plasticity angle**

In our first attempt at correction, we took the final checkpoints of the shortcut and control agent and gave them 200k training steps to correct themselves and learn the genuine task. They were also given an entropy bonus to boost the correction. What we found was surprising: despite our large training budget, **both the agents** struggled to correct themselves.

Playing back their Pygame visualizations, we realized that despite constant feedback that their original strategy of going south and claiming near the (4,0) square was not fit for the new task, the agents continued to do just that. Only towards the last 20% of the training run, did we start to see the agents exploring other regions of the board. By then, however, it was too late

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Correction after 200,000 training steps"><source src="/videos/reward-hacking/correction-200k.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Correction after 200,000 training steps</figcaption>
</figure>

Our first debugging attempt was first to simply increase the training budget and see if both agents learn given more time. Since they began to navigate and explore the board towards the end of the 200k-step run, we thought giving them more chance to explore and randomly encounter rewards would reinforce the right strategy. The results of our 500k budget run are once again…surprising. 

<div class="video-pair">
<figure>
  <img src="/images/projects/reward-hacking/article-1.png" alt="Learning curves from the 500k-step correction run">
  <figcaption>Learning curves from the 500k-step correction run</figcaption>
</figure>
<figure>
  <video controls muted preload="metadata" playsinline aria-label="Correction after 500,000 training steps"><source src="/videos/reward-hacking/correction-500k.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Correction after 500,000 training steps</figcaption>
</figure>
</div>

Clearly the agent has now learnt to explore the board, but it still doesn’t seem to be **claiming the beacon.** So we then started to analyze the actions taken by our agent. Clearly, it was getting to the beacon, *was it just not choosing the ‘claim’ action?* The results were staggering. From ~37500 actions, our agent only tried to claim 131 times! Moreso, 114 of those claim calls came in the final 10% of the training run. *Our agents never learnt to actually finish the task.* A large reason for this was the reward design. Every wrong claim incurred a *-0.03* penalty, while choosing any other action resulted in just a *-0.01* penalty. Thus, our policy was super afraid to try claiming and being wrong. Changing the reward to stop penalizing wrong claims more than normal actions did the job. 

But the problem above also points to a **deeper problem.** The model checkpoints we used, from the end of the baseline training run, had learnt to navigate to the (4,0) square and claim only near the bottom of the board. But what’s worse is, they were **extremely confident** in that strategy—since that is what was optimal for the fixed beacon and shortcut tasks. 

*What would happen if we tried to correct earlier checkpoints, which were competent at their respective tasks but significantly **less confident** of their actions? Would they learn faster?* The reliably competent checkpoints successfully corrected itself within just 100k steps, despite the negative penalty for wrong claims. Analyzing the action distribution and board visualizations, it became obvious that this policy not only started exploring the board from earlier in the training, but it was also **not afraid to claim**—even wrongly so—from other squares apart from (4, 0). This is owed largely to this checkpoint’s lower confidence in its learned behavior.

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Correction from the reliably competent checkpoint"><source src="/videos/reward-hacking/correction-reliable.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Correction from the reliably competent checkpoint</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/article-2.png" alt="S-variable learning curves starting from earlier checkpoints">
  <figcaption>S-variable’s learning curves starting from earlier checkpoints</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/article-3.png" alt="G-fixed learning curves starting from earlier checkpoints">
  <figcaption>G-fixed’s learning curves starting from earlier checkpoints</figcaption>
</figure>

This result is also corroborated by [earlier studies on plasticity](https://proceedings.mlr.press/v202/lyle23b.html). It seems to be the case that trained networks are less adaptable and struggle to pick up new tasks. So, for the remainder of our experiments, we decided to use the earlier but competent checkpoints to measure learning outcomes. 

**Findings from RQ 1**

Running the correction across all our seeds, we found that the G-fixed agent learned to correct itself much faster than the shortcut agent. As seen from the figure above, the G-fixed agent become ‘reliably competent’ at the task, with `success_rate ≥ 0.9`  & `len(model_path) - len(shortest_path) ≤ 2` , near 110k training steps on average across all seeds. Meanwhile, the shortcut agent corrected itself much later at around 160k steps. Not only that, the shortcut agent also performed much worse at other metrics. Its average return collected was much lower, how well it generalized to unseen locations (as seen by the evaluation success rate) was worse, and also, it struggled to find shortest paths to the beacon (as evidenced by the average episode length being higher). Clearly, having a history of exploiting the shortcut led to slower learning.

## Q2 - What about acquiring new tasks?

### Findings

### Next Questions

[https://x.com/meowkoteeq/status/1750232565737238756](https://x.com/meowkoteeq/status/1750232565737238756)

**Note this for later study:** Reward hacking ultimately occurs due to the environment’s flawed reward system. A pertinent question could be about whether or not agents learn to hack even without rewards….See:

[https://x.com/owainevans_uk/status/2119520230829641811?a=rzPo1E5-21aX3626YBk&s=46](https://x.com/owainevans_uk/status/2119520230829641811?a=rzPo1E5-21aX3626YBk&s=46)

*catastrophic forgetting*: forgetting means learning something new damages old knowledge; loss of plasticity means the network struggles to learn the new thing in the first place.