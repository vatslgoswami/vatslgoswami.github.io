---
title: How reward hacking affects learning
year: 2026
kind: Research
summary: A proposed study of whether learning a reward shortcut affects an agent's ability to correct its behavior or learn a new task.
tags: [AI safety, Reinforcement learning]
links:
  - { label: GitHub, url: "https://github.com/vatslgoswami/hacking-affects-learning" }
featured: true
detail: true
---

This project asks whether training an agent to exploit a reward shortcut makes it harder to learn the intended task later. If a disadvantage appears, is it because the shortcut kept the agent from getting useful practice, or does a difference remain even when agents receive the same useful examples?

The proposed study uses a small gridworld where an agent can receive a completion reward without reaching its intended goal. It compares that training history with agents given legitimate navigation practice and an untrained reference. After the faulty completion checker is repaired, separate experiments will measure correction on the original navigation task and learning a new package-delivery task.

The study also plans to compare learning from identical demonstrations and from guided practice. These diagnostics may help explain any differences in recovery, while recognizing that a small gridworld cannot establish how larger language models would behave.

**Status:** Research proposal; experiments and results are forthcoming. Shortcut training could hinder, help, or have no detectable effect on later learning.
