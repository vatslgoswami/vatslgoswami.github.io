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
wordCount: 4999
detail: true
---
Note: I use the words ‘agent’ and ‘policy’ almost interchangeably across this blog. RL folks, please pardon me xD.

## First, a real world scenario

Every once in a while, my X feed gets filled with memes about how ChatGPT can solve Millennium Prize problems and yet can’t answer a simple question in one sentence (e.g., [this thread](https://x.com/meowkoteeq/status/1750232565737238756)).  And if you’re someone who uses AI regularly, I’m sure you’ve also experienced situations where agents just can’t seem to understand what you want from them. Frustrating.  Below is a non-exhaustive list of some *more* *serious* cases of misaligned behavior from ‘smart’ AI systems:

- LLMs giving extremely sycophantic responses. Or hallucinating facts to come off as confident.
- A coding agent completing software tasks by changing regression tests instead of actually solving the bug.
- A robot tasked with keeping your home clean starting to continuously clear the trash and bring it back in — only to clear it out again in an endless loop.
- An agent trained to play CoastRunners, a boat racing game, going round and round in circles collecting points instead of trying to win the race (see: OpenAI’s [blog](https://openai.com/index/faulty-reward-functions/) on this)

Before we understand why the above situations arise, here’s the background knowledge I assume:

- Basic familiarity with LLM training stages: Pre-Training → SFT → Post-Training techniques.
- An understanding of the RL setup: environment, agent, actions, rewards, etc.
- Intuition about how some of the modern RL learning algorithms like PPO, GRPO etc. work.

If you’re not fully comfortable with some of the topics above, or need a quick refresher, here is a quick read to get you ready: [*A vision researcher’s guide to PPO and GRPO.*](https://yugeten.github.io/posts/2025/01/ppogrpo/)

Back to the topic. Training an agent to achieve an objective using Reinforcement Learning depends largely on how you define the reward function, and more importantly how you check if the agent completed your task genuinely. With math or coding tasks, you could use deterministic checkers to verify whether an LLMs solution is correct and reward correct answers. For more subjective tasks, like getting an LLM to be more ‘helpful’, you could train a reward-model on real human feedback and optimize LLM output to match what kinds of responses humans like.

The problem arises when agents find loopholes to game the reward system. This can be very hard to avoid.  How do you accurately determine when an task is completed ‘genuinely’, or as intended? Conversely, how do you catch when an agent is faking a task just to earn reward?

Take the situations we listed above. Here are some ways in which an AI agent may have learnt to exploit even a seemingly well-defined reward:

- <u>Sycophantic & overly-confident responses:</u> An LLM midst post-training realizes that human annotators prefer (and thus give it reward) when its responses sound extremely confident—even if factually incorrect—or when responses are super in-agreement with the user.
- <u>Coding Agent:</u>  The agent realizes that it gains a reward when it passes some tests. Thus instead of learning to fix the bugs, it just learns to edit the tests.
- <u>Househelp Robot:</u> The robot realizes that clearing out the trash is a high-reward action and learns to continuously bring the trash in and out to maximize reward. This is clearly not what the designers of the robot would’ve intended!
- You get the gist…

Learning algorithms are just optimizing a policy’s weights to take actions/trajectories that maximize expected rewards; the utility $U(\theta)$, i.e. how good a model’s weights are, measures exactly this quantity.

$$
U(\theta) = \sum_\tau P(\tau|\theta)R(\tau)
$$

If there is a shortcut that is easy for the policy to learn/stumble upon, and it gets rewarded for that faulty shortcut, it is natural that the policy will learn to exploit that action. Figuring out how to define our environments and reward functions to actually teach the agent what we want it to learn is one of the core challenges of RL. (To learn more about the challenges in RLHF, do check out [this amazing talk](https://www.youtube.com/watch?v=aM6rsVZ1GAU) by Stephen Casper at EAGxBoston.) The tendency of models to exploit loopholes in the reward function is known as Reward Hacking. In this blog, we’ll be looking at how reward hacking in models can affect subsequent learning outcomes, like learning a new task, even after the reward function has been fixed.

<figure>
  <img src="/images/projects/reward-hacking/final/Screenshot_2026-11-09_at_3.23.52_AM.png" alt="Illustration about designing rewards and incentives">
  <figcaption>Source: <a href="https://www.evilaicartoons.com/archive/design-good-carrots-and-sticks">Evil AI Cartoons</a></figcaption>
</figure>

---

## Motivation

Let’s motivate this study by diving a bit deeper into the coding agent example. Consider a coding agent that is put in an RL environment where it has to complete software-engineering tasks. The agent gets a reward if its code passes an automated verifier’s checks. Now, assume that the verifier only tests the agent’s code on certain inputs, and the agent learns to map what inputs it’s getting tested on with their expected output values. Then, it can easily hardcode the values of the function, pass the verifier’s check, and get the reward: i.e., the agent learns to reward hack instead of completing the intended task.

Let’s say the company training the agent figures out that this is happening and fixes their automated verifier. The reward system has now been fixed; the agent will only receive the reward if it writes the intended functions. At this juncture, before restarting training, the company will need to decide whether they want to continue training from the checkpoint that learnt to hack or restart training from scratch. *How does reward hacking affect subsequent learning outcomes in agents?* Making progress on this question is what motivates this study.

---

## Experimental Setup

The RL environment for this study was largely inspired by [Leike et.al’s work](https://arxiv.org/html/1711.09883) on AI safety gridworlds. Gridworlds were the perfect setup since they could run easily on my CPU and their simplistic design made them the perfect playing ground for designing multiple tasks and interventions.

**Setup:**

Here is an overview of the experiment setup. All the tasks are built around navigating a 7x7 Gridworld. The agent has 6 actions: north, east, south, west, claim, and pickup. We have finite-horizon tasks which are to be completed within 30 actions per episode. There are two different kinds tasks designed for our agent:

1. `claim-beacon:` Go to the beacon’s location and claim it to get a reward. Here, there are two variations: one with the beacon position **fixed** at a cell, and another where the beacon spawns at a **random position** on the board.
2. `pickup-and-deliver:` The agent has to go to a location, pickup a package, and deliver it to a destination to claim the reward. Here, the package and destination keep spawning at random positions each episode.

**Reward System:**

For `claim-beacon` tasks, the agent receives:

- $+1$ for claiming the beacon
- $-0.03$ for wrong claims, where `agent.position ≠ beacon.position`
- $-0.01$ for any action taken

Clearly, the rewards are designed to incentivize the agent to navigate to the beacon’s location via the shortest path possible and claim only if it’s confident that it has reached the correct location.

For `pickup-and-deliver` tasks:

For this task, we originally planned to keep the following reward system:

- $+1$ for for claiming/delivering the package at the correct delivery location
- $-0.03$ for wrong claims
- $-0.01$ for any action
- No reward for picking up the package

However, within a few smoke tests itself, we realized that this reward system won’t work. The `pickup-and-deliver` task is sequential and needs the agent to learn both how to navigate to the first pickup location and then correctly deliver the package to the destination. Only rewarding the delivery stage meant that nothing was incentivizing our policy to learn to complete the pickup successfully first. Add to that the fact that rewards were already too sparse—since it’s very unlikely that our initialized policy will be able to complete the task within the 30 action limit. Both these problems resulted in our smoke-test failing: the policy was not learning despite a large training budget.

So, we came up with a revised reward function as follows:

- $+0.75$  for successful delivery
- $+0.25$ for successful pickup
- Wrong claims and actions retain their $-0.03$ and $-0.01$ rewards.

As we’ll see later on, this two-step reward, coupled with a curriculum and a entropy bonus to encourage exploration, tackled the sparse reward problem and also allowed the policy to gain more useful navigation experiences in the initial phases of training.

<u>The Hidden Shortcut</u>

Inside the `claim-beacon` task, we introduced a hidden shortcut at the bottom of the grid. Going to this cell and claiming would trigger a proxy reward of $+1$ for our agent. This is the loophole that our agent can learn to exploit.

<figure>
  <img src="/images/projects/reward-hacking/final/Screenshot_2026-11-09_at_6.27.23_AM.png" alt="Pygame visualization of the grid with a fixed shortcut at (4,0)">
  <figcaption>Pygame visualization of the grid with a fixed shortcut at (4,0)</figcaption>
</figure>

**Training Four Baseline Agents**

Based on the different tasks above, we train 4 PPO agents on different histories; one for each task. The agent was a 2-layer MLP with the action and value heads sharing the internal trunk. We chose a shared trunk to reduce computational load.

<figure>
  <img src="/images/projects/reward-hacking/final/image.png" alt="Our policy network architecture">
  <figcaption>Our policy network architecture</figcaption>
</figure>

A little bit about the naming convention used for each of our baseline agents. G stands for *genuine*: denoting the agents that trained in an environment without any shortcuts. S stands for *shortcut*: agents which had the shortcut made available to them.

1. <u>G-Fixed:</u> This agent trains on a narrow task similar to our shortcut agent. The beacon position is fixed at $(4,0)$ which matches the shortcut square.
2. <u>G-variable:</u> This agent trains with a beacon that spawns randomly anywhere on the board, apart from the $(4,0)$ cell. This was done so that G-variable’s beacon comes from the same distribution as our shortcut agent.
3. <u>S-variable:</u> This agent trains with a variable beacon position — however, the proxy reward at square $(4,0)$ is made available to this agent.
4. <u>G-pickup-deliver</u>: This is a fresh, control agent trained from scratch on the `pickup-and-deliver` task.

**Run Configuration**

- The training budgets for the first  $3$ agents was $75,000$ steps. Since the `pickup-and-deliver` task was much more complex, we gave the last agent  $300,000$ steps. Both these budgets were decided based on initial trial runs.
- All agents shared the same entropy-decay schedule: the entropy coefficient started off at $0.1$ and decayed linearly to $0.01$ by the time $35\%$ of the training budget was reached. It stayed constant at $0.01$ afterward. This was chosen to allow our policies to explore more initially and only become more confident in their actions after they have spent significant time figuring out patterns of the task and the optimal strategy.
- The PPO rollout size was $2048$ steps.

**Evaluation & Checkpointing**

After every rollout, we ran an evaluation script to check whether the model was actually learning and improving at the task.

- $40\%$ of the evaluation episodes consisted positions which were completely new and unseen to the agent. This was done by creating a train-test split of positions on the board, and allowed us to see if the agent generalizes well, i.e. whether it was actually learning to navigate the grid or just memorizing positions.
- The evaluations ran $25$ episodes and choose actions stochastically. For each batch of $25$ episodes, the evaluations returned the following metrics:

→ *success rate:* $\dfrac{\#\text{ successful beacon claims}}{25}$

→ *avg return per episode:* $\dfrac{\sum_{i=1}^{25} r_i}{25}$, where $r_i$ is the reward gained in episode $i$.

→ *avg episode length:* $\dfrac{\sum_{i=1}^{25} \operatorname{len}(e_i)}{25}$, i.e how many actions it took the agent to complete the task on avg (or hit the 30-action limit).

We saved model checkpoints every $10,000$ steps. Along with that, we also checkpointed the first rollout after which the model became *reliably competent* at the task. Our criteria for competence required the model to meet two conditions: one, that it should succeed enough (`success_rate ≥ 0.9` ); two, that it should do so in an efficient manner (`|len(model_path)-len(shortest_path)| ≤ 2`). If our model met this criteria in two successive evaluations, then it was saved as the *reliably competent* checkpoint. In simple words, this is our way of finding the point during training at which the model became ‘good’ at its task.

**Research Questions:**

Using the setup and tasks designed above, we tried to answer the following questions:

1. Once the shortcut is removed, is the S-variable agent slower to learn the `claim-beacon` task than a control who was also trained initially on a fixed, narrow task (G-fixed agent)? *This can be thought of as question of how easily the agent is able to correct itself.*
2. Is the S-variable agent able acquire a new task (`pickup-and-deliver`) just as well as other controls? *This can be thought of as a question of knowledge transfer.*
3. Can the hindered learning outcomes be attributed to internal representations of the agent or simply to a lack of useful experience navigating the board?

---

## Baseline Training Results

Before we could start running experiments and test each of our research questions, we had to train our baseline agents. Each of the agents learnt to reliably complete their respective tasks well within the training budget allotted. By the end of the run, they all converged to finding the shortest path to the beacon. This holds true for all the different RNG seeds we tried.

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Genuine variable agent"><source src="/videos/reward-hacking/genuine-variable-recording.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>G-variable agent training run</figcaption>
</figure>

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Shortcut variable agent"><source src="/videos/reward-hacking/hacky-variable-recording.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>S-variable agent training run</figcaption>
</figure>

The only major setback we faced in this phase was in teaching a fresh agent how to learn the `pickup-and-deliver` task. Even with the two-step reward, the agent was struggling to make progress on the task. It learnt to pickup the package, but since it struggled to reach the destination and claim the additional $+0.75$ reward, the agent did not get enough learning signal. Ultimately, it just settled for the $+0.25$ reward from the pickup, instead of trying to find the delivery location.

<u>Introducing a curriculum</u>

It was very tough for a random policy to complete tasks, and so the rewards were too sparse. A solution we found to mitigate this was to introduce a curriculum and train the agent in reverse (credit: [Reverse Curriculum Generation for RL](https://arxiv.org/html/1707.05300)). This was done by starting a fraction of each rollout from an assisted state, where the package is already picked up and the agent is placed only $2$ squares away from the delivery destination. We trained the agent on $500k$ steps, starting off with $50\%$ of the training episodes being assisted and linearly decayed the assistance to zero by $300k$ steps. With this strong assistance, the agent finally converged! But as you can see from the figure below, the agent was still struggling to gain good learning signal for the first $250k$ steps.

The agent would initially pickup the package and try to explore all the squares within a distance of $2$, even when the destination was located furhter away. Other times, it would directly go to the destination instead of navigating to the pickup location first. We needed to steer the agent to venture further away from the pickup point and increase its chances of stumbling upon the massive reward received upon successful delivery.

<figure>
  <img src="/images/projects/reward-hacking/final/image-1.png" alt="Learning curves for pickup-and-deliver with reverse curriculum">
  <figcaption>Learning curves for pickup-and-deliver with reverse curriculum</figcaption>
</figure>

<u>Adding entropy bonus</u>

That’s when we decided to add an entropy bonus to the agents. As you can see in the figures below, adding an entropy bonus encouraged the agent to continue to explore more of the board, by keeping its policy distribution uniform instead of becoming confident early on. The agent now saw the full $+1$ reward more often earlier and slowly the policy converged to navigating to both the pickup and delivery locations. The agent now learnt the optimal, shortest path strategy in just $300k$  steps!

<figure>
  <img src="/images/projects/reward-hacking/final/image-2.png" alt="Learning Curves for Curriculum + Entropy Bonus on pickup-and-deliver task">
  <figcaption>Learning Curves for Curriculum + Entropy Bonus on pickup-and-deliver task</figcaption>
</figure>

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Pickup and delivery with entropy bonus"><source src="/videos/reward-hacking/pickup-deliver-with-entropy.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Pickup and delivery with entropy bonus</figcaption>
</figure>

---

## The Correction Task

We now had everything we needed to answer our first question. To test whether the S-variable agent could correct its behavior, we removed the shortcut reward and continued training it on the `claim-beacon` task. The agent could now earn a positive reward only by navigating to and claiming the beacon.

We compared the shortcut agent’s learning speed with the **G-fixed** agent, which had initially trained on a beacon fixed at $(4,0)$—the same location as the shortcut. This makes for a useful control because both these agents had learned a narrow behavior of navigating to the same square again and again. The only difference is that G-fixed had learned it by completing the intended task, while S-variable had learned to exploit the shortcut. If the latter struggles to learn the navigation task, then this suggests that *which square the agent learned to visit* does not fully explain its slow learning; shortcut history makes correction harder, at least in our Gridworld setup. We will later use interventions to see which aspects of their training histories create this gap.

**A small digression: the plasticity angle**

In our first attempt at correction, we took the final checkpoints of the shortcut and control agent and gave them $200k$ training steps to correct themselves and learn the genuine beacon task. They were also given an entropy bonus to boost the correction. What we found was surprising: despite our large training budget, **both the agents** struggled to correct themselves.

Playing back their Pygame visualizations, we realized that despite constant feedback that their original strategy of going south and claiming near the $(4,0)$ square was not fit for the new task, the agents continued to do just that. Only towards the last 20% of the training run, did we start to see the agents exploring other regions of the board.

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Correction after 200,000 training steps"><source src="/videos/reward-hacking/correction-200k.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Correction after 200,000 training steps</figcaption>
</figure>

Our first debugging attempt was first to simply increase the training budget and see if both agents learn given more time. Since they began to navigate and explore the board towards the end of the $200k$-step run, we thought giving them more chance to explore and encounter more rewards would reinforce the right strategy. The results of our $500k$ budget run are once again…surprising.

<figure>
  <img src="/images/projects/reward-hacking/final/image-3.png" alt="Learning curves from the 500,000-step correction run">
  <figcaption>Learning curves from the 500,000-step correction run</figcaption>
</figure>

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Correction after 500,000 training steps"><source src="/videos/reward-hacking/correction-500k.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Correction after 500,000 training steps</figcaption>
</figure>

Clearly the agent had now learnt to explore the board, but it still doesn’t seem to be **claiming the beacon.** We then started to analyze the actions taken by our agent. It was getting to the beacon, *was it just not choosing the ‘claim’ action?* The results were staggering. From $\sim37500$  actions, our agent only tried to claim $131$ times! Moreso, $114$ of those claim calls came in the final $10\%$ of the training run. *Our agents never learnt to actually finish the task.* A large reason for this was the reward design. Every wrong claim incurred a $-0.03$ penalty, while choosing any other action resulted in just a $-0.01$ penalty. Thus, our policy was **super afraid** to try claiming and being wrong. Changing the reward to stop penalizing wrong claims more than normal actions fixed the training loop.

But the issue above also points to a deeper problem. The model checkpoints we used, which came from the end of the baseline training run, had learnt to navigate to the $(4,0)$ square and claim only near the bottom of the board. But what’s worse is, they were also **extremely confident** in that strategy—since that is what was optimal for the fixed-beacon and shortcut tasks.

*What would happen if we tried to run the correction task on earlier checkpoints, which were competent at their respective tasks but significantly **less confident** of their actions? Would they learn faster?* The reliably competent checkpoints successfully corrected themselves within just $100k$ steps, despite keeping the large penalty for wrong claims. Analyzing the action distribution and board visualizations, it became obvious that this policy not only learned quicker that it should explore other regions of the board, but it was also not afraid to claim—even wrongly so—from other squares apart from $(4, 0)$. This is owed largely to this checkpoint’s lower confidence in its learned behavior.

<figure>
  <video controls muted preload="metadata" playsinline aria-label="Correction from the reliably competent checkpoint"><source src="/videos/reward-hacking/correction-reliable.mp4" type="video/mp4">Your browser does not support video playback.</video>
  <figcaption>Correction task starting from the reliably competent checkpoint</figcaption>
</figure>

This result is also corroborated by [earlier studies on plasticity](https://proceedings.mlr.press/v202/lyle23b.html). It seems to be the case that trained networks are less adaptable and struggle to pick up new tasks. We too found that this difference in learning between the final and earlier checkpoint **held strongly** throughout every other task and intervention we introduced in this study. You can see more plots for the same on my [Github repo](https://github.com/vatslgoswami/hacking-affects-learning) under `/final_plots`. For the remainder of this blog, we’ll be looking at the results that used the earlier, more flexible checkpoints of our baseline models.

<figure>
  <img src="/images/projects/reward-hacking/final/image-4.png" alt="Fig: S-variable’s learning curves starting from the ‘reliably competent’ checkpoints">
  <figcaption>S-variable’s learning curves starting from the ‘reliably competent’ checkpoints</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/final/image-5.png" alt="Fig: G-fixed’s learning curves starting from ‘reliably competent’ checkpoints">
  <figcaption>G-fixed’s learning curves starting from ‘reliably competent’ checkpoints</figcaption>
</figure>

**Findings from RQ 1**

Running the correction across all our seeds, we found that the G-fixed agent learned to correct itself much faster than the shortcut agent. As seen from the figure above, the G-fixed agent become ‘reliably competent’ at the `claim-beacon` task at around $110k$  training steps. Meanwhile, the shortcut agent corrected itself much later at around $160k$ steps. Not only that, the shortcut agent also performed much worse at other metrics. Its average return per episode was much lower, how well it generalized to unseen locations was worse (as seen by the evaluation success rate), and also, it struggled to find shortest paths to the beacon (as evidenced by the average episode length being higher). Clearly, having a history of exploiting the shortcut led to slower learning.

---

## Acquiring New Tasks

We now test how easily the shortcut agent acquires a new task: `pickup-and-deliver`. This time also, we benchmark how S-variable performs against G-fixed.

**Findings:**

The `pickup-and-deliver` task was much more complex and both the agents struggled to learn it despite having a reverse curriculum for the delivery stage. In $300k$  training steps, both agents were able to learn only the pickup stage of the task and their average returns flattened at around $-0.05$.

This result perfectly sets up the interventions we introduce in the next step. We’ve seen by now that the shortcut agent was much slower to correct itself, and also struggled to pick up a new task. Can we design interventions to test why this happens?

---

## Slow learning…but why?

We have now established that in both our tasks, correction and knowledge transfer, a history of shortcuts hindered the agent’s ability to learn. This could be a result of two root causes.

1. The faulty environment resulted in the policy only learning to navigate to the shortcut square. Starting with such a policy on the `claim-beacon` and `pickup-and-deliver` tasks meant that early on in the training run, the agent struggled to gain any **useful experiences** on what kinds of actions produce good reward.
2. Another possible reason could be that the choosing to exploit the shortcut created **internal representations** (how the network understands the features of the board, patterns of the problem at hand, etc.) of the task that were incoherent with completing the genuine tasks assigned to the agent.

We designed two interventions to isolate both these probable causes.

**Intervention 1: Supervised Imitation**

Instead of directly running PPO on our baseline checkpoints, this time, we first give them a supervised learning task where they imitate an optimal policy. This was done to give the agents useful experience, by teaching them what kinds of actions they should take from certain positions. This fixes the cold-start problem caused by agents spending a large chunk of initial training time following their old strategies, which don’t translate well to the new task. For the beacon-task, we used $100$ distinct board configurations: this corresponds to only $4\%$ of all possible $(agent, beacon)$ positions. For the `pickup-and-deliver` task, we only trained the agent on the delivery stage, again with $100$ distinct positions: only $0.09\%$ of all $(agent, package, destination)$ positions. We trained for $1000$ updates with batches of $64$.

**Findings:**

Giving our agents even a little useful experience went a long way in both tasks. In the beacon task, averaging across all seeds, our G-fixed agent now became reliably competent at the beacon task in just $28k$ steps compared to the $118k$ steps it took without the intervention. A similar drastic reduction can be seen with the shortcut agent.

<u>G-fixed agent on claim-beacon task</u>

<figure>
  <img src="/images/projects/reward-hacking/final/image-6.png" alt="Without imitation">
  <figcaption>Without imitation</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/final/image-7.png" alt="With imitation">
  <figcaption>With imitation</figcaption>
</figure>

<u>Shortcut-agent on claim-beacon task</u>

<figure>
  <img src="/images/projects/reward-hacking/final/image-8.png" alt="Without imitation">
  <figcaption>Without imitation</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/final/image-9.png" alt="With imitation">
  <figcaption>With imitation</figcaption>
</figure>

In the previous section, we had seen both our agents fail to learn the more complex `pickup-and-deliver` task—even with a reverse curriculum for teaching the delivery stage. However, with our imitation intervention, they both not only begin to converge to the optimal policy but also do so in the **same time** it took a fresh agent. Seems like a little imitation learning can give you massive gains!

<u>G-fixed on pickup-and-deliver task</u>

<figure>
  <img src="/images/projects/reward-hacking/final/image-10.png" alt="Without imitation — policy does not learn the task">
  <figcaption>Without imitation — policy does not learn the task</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/final/image-11.png" alt="With imitation — policy learns the task and that too very quickly">
  <figcaption>With imitation — policy learns the task and that too very quickly</figcaption>
</figure>

<u>Shortcut agent on pickup-and-deliver:</u>

<figure>
  <img src="/images/projects/reward-hacking/final/image-12.png" alt="Without imitation">
  <figcaption>Without imitation</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/final/image-13.png" alt="With imitation">
  <figcaption>With imitation</figcaption>
</figure>

This intervention shows us that imitation learning can be a powerful way to teach policies patterns of the intended task quickly, so that they can start to pickup useful experience and learning signals right from the beginning of their training runs — resulting in a much more efficient RL phase. More importantly for this study,  the findings above support the hypothesis that the slow learning of our S-variable agent can be largely attributed to its lack, or rather inability, to perform actions that gave it useful learning experiences.

**Intervention 2: Action-head reset**

Our second intervention is designed to test if the internal representations of our S-variable agent played a role in slowing down its learning. We have seen that after baseline training, the agent strongly prefers trajectories that take it to the shortcut square and claim there. With the shortcut now removed from the environment, those action preferences make it unlikely for the agent to reach the randomly spawned beacon and complete the intended task. However, what if the **action preferences were reset** — so that at each position, what the action the agent chooses is random. If our agent still struggles to pickup the new task, it could suggest that its **internal understanding**, developed during the hacking phase, of the features and patterns of the grid are affecting its ability to learn the new tasks.

To do this, we re-initialize the agent’s action head while keeping the shared trunk the same. S-variable now starts the correction and knowledge transfer tasks with features it learned before but new action preferences.

**Findings:**

<figure>
  <img src="/images/projects/reward-hacking/final/image-14.png" alt="Action-head reset on claim-beacon">
  <figcaption>Resetting S-variable's action head on claim-beacon task</figcaption>
</figure>

<figure>
  <img src="/images/projects/reward-hacking/final/image-15.png" alt="Action-head reset on pickup-and-deliver">
  <figcaption>Resetting S-variable's action-head on the pickup-and-deliver task</figcaption>
</figure>

Just as before, this intervention also drastically improved the learning outcomes in our shortcut agent, supporting our hypothesis that the agent’s **confident** action preferences were the major driver of slow learning (and not the internal features). The separation isn’t perfect, though. Resetting the action head changes how the agent explores, but the shared trunk also continues to update during training. So an improvement caused by resetting the action head doesn’t prove that the action preferences were the only problem, or that the trunk’s internal representations are harmless. However, it does give us an important intervention to correct networks **without re-training** from scratch.

---

## Conclusion

Our study of Gridworlds leads to many interesting observations about how a history of shortcut learning may have a negative impact on subsequent learning outcomes. We first saw that as models train to optimize for a task, they become more and more confident in their solution and lose the ability to adapt to newer, adjacent tasks. Next, we established that previous, hacky behaviour affects an agent’s ability to correct itself. Moreover, it also slows down learning in newer tasks. Finally, we saw two interventions that helped improve learning outcomes in our agents; both worked by tackling the agents’ action preferences, giving it useful learning signal earlier in the training.

I’d like to add that all of the above findings are only within the context of our Gridworld setup. None of them automatically generalize to bigger systems like LLMs, and nothing done in this study posits such a causal link. However, the interventions used here can easily be tested on deeper networks and may even prove to be powerful methods to correct ‘bad’ behavior. A lot of this, for example imitation learning, is already done in the SFT phase of post-training. This raises an interesting question: *why do LLMs still end up reward hacking, even on tasks (such as coding) where they have received examples of what a good solution looks like?* A much more interesting question would be to think from an Alignment lens…*What does a ‘good’ example of aligned behavior (that we can give an LLM to imitate) look like?* I don’t think this problem has been solved yet. Such an inquiry doesn’t have to be so broad either. For instance, we could just fixate on one behavior that we may want to correct in an LLM, like being sycophantic. What kind of responses could we provide for imitation so that the LLM learns to agree when it thinks the user is right and yet not end up becoming overly critical of user responses.

During the course of this study, a couple other questions came to my mind, which I hope to get to some time in the near future.

1. I stumbled upon [this thread](https://x.com/owainevans_uk/status/2119520230829641811?a=rzPo1E5-21aX3626YBk&s=46) by Owain Evans while writing this report and it really made me wonder *whether agents would learn to hack even if they weren’t rewarded for it.* My hunch would be no, unless they have a hint about why it might be in their benefit to do so.
2. *Can reward hacking lead to catastrophic forgetting?* That is, can an agent that begins to exploit its way through tasks lead it to lose older capabilities?

That’s all for now! Thanks for sticking around: I know it’s been a long read. If you have any feedback or would want to collaborate on any project, please feel free to [reach out](mailto:vatslgoswami@gmail.com)!

Until next time,  
-Vatsl.

*"Wisdom begins in wonder."*
