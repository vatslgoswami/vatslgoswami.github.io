---
title: Does Reward Hacking Hinder Subsequent Learning?
year: 2026
kind: Research
summary: A study of how reward hacking affects an agent's ability to learn after a faulty reward mechanism is repaired.
tags: [RL, Learning Dynamics, Alignment]
links:
  - { label: GitHub, url: "https://github.com/vatslgoswami/hacking-affects-learning" }
featured: true
detail: true
---

An agent trained against a flawed reward mechanism can learn to achieve a high score without completing the task its designers intended. This project investigated what happens *after* that flaw is fixed: whether the agent learns the intended task as readily as other agents, and whether its earlier training affects its ability to acquire a different skill.

## Motivation

Consider a coding agent rewarded for passing a test suite. Instead of fixing a bug, it changes the tests so they pass. The reward signal registers success, but the underlying problem remains. Even after the test suite is repaired, the agent has spent training time practicing a shortcut rather than debugging. Its prior training may also have shaped action preferences or internal features that influence how it learns from later feedback.

This example motivates a distinction between two possible sources of difficulty: **missing useful practice**, and effects of the **training history retained in the policy**. The project studied that distinction in a controlled gridworld, where intended success and the reward signal could be measured separately. The coding-agent example is an analogy; the experiments did not train or evaluate coding agents.

## A toy model of a faulty checker

I built a gridworld in which an agent navigated to a visible beacon and claimed completion. Genuine success required reaching the beacon before making the claim. During shortcut training, a faulty checker also rewarded a claim at a fixed, unmarked square, even when the beacon was elsewhere. The agent could receive the same completion reward without doing the intended task.

The environment tracked genuine completion separately from accepted false claims, so a high reward did not automatically count as task success. The shortcut was not always the shortest route: for some beacon locations, reaching the real goal was more efficient. This made the agent's behavior dependent on the opportunity presented by each episode.

## Comparing training histories

I trained PPO agents under four histories:

- **Shortcut access:** a variable beacon and a faulty checker that accepted claims at the fixed shortcut.
- **Narrow legitimate practice:** a fixed beacon at the shortcut location, with a correct checker.
- **Varied navigation practice:** a variable beacon and a correct checker.
- **No prior training:** a fresh agent used as a reference.

The narrow-practice comparison gave agents experience with a similar repeated route without rewarding false completion. The varied-practice comparison provided broader navigation experience. Together with the fresh reference, these histories helped distinguish the effects of shortcut training from the benefits of useful practice and from effects of a narrow routine more generally.

## Correction and transfer

After source training, I repaired the checker and evaluated separate branches from the saved agents. In **correction**, agents learned to reach variable beacons under accurate checking. In **transfer**, they learned a new pickup-and-delivery task: reach a package, pick it up, carry it to a destination, and claim completion. The two branches started independently from the source checkpoints, so delivery learning did not depend on first completing correction training.

I measured genuine task success across training checkpoints, rather than treating reward alone as evidence that an agent had learned the intended behavior. Comparing the learning curves across histories addressed both recovery on the original task and transfer to a new sequential skill.

## Testing explanations

I used shared demonstrations, policy-head resets, and guided practice to investigate why learning might differ across histories. In the common-data diagnostic, agents received the same useful examples, allowing their learning to be compared when access to demonstrations was held fixed. Comparing intact policies with policies given a common fresh action head helped separate the role of existing action preferences from effects associated with retained policy features.

Guided practice provided agents with more opportunities to encounter useful task experience during reinforcement learning. Its effects were evaluated on the original, unassisted task. This intervention changed both the experience agents encountered and their training updates, so it was informative about recovery without uniquely identifying a mechanism.

## Scope

The experiments isolated a particular reward-checking flaw in a small gridworld. Differences between training histories reflect the combined effects of their experiences and updates; they do not isolate a universal effect of reward hacking or establish that an agent understood it was exploiting a flaw. The results of this toy setting do not by themselves establish how language models or coding agents behave.
