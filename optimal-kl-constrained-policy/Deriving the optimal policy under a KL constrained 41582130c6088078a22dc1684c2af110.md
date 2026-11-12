# Deriving the optimal policy under a KL constrained objective

## Introduction

LLMs are pre-trained on large, internet scale data. Naturally, they inherit a lot of biases, harmful behaviors, and misleading information. One of the goals of post-training stages is to mitigate some of these bias and improve model behavior on more abstract objectives like “helpfulness” or “preventing abusive responses”. To do this, we use [RLHF](https://huggingface.co/blog/rlhf) — mainly because that’s just the best we currently have. Over the years, researchers have uncovered loads of problems with this technique (see [Open Problems with RLHF](https://liralab.usc.edu/pdfs/publications/casper2023open.pdf) by Casper, et. al). Some of these include: 

- Challenges with using human evaluators to collect preference data. Annotators may themselves have harmful beliefs, fall prey to hidden biases (like the tendency to like syncopathic responses), be unable to verify factually incorrect responses, etc.
- There is doubt whether RLHF can lead to **robust alignment** of models. Research shows that RLHF may improve model behavior in familiar settings but undesirable behaviors can reappear under certain interventions like fine-tuning (see [Fine-Tuning Models Compromises Safety](https://arxiv.org/abs/2310.03693)).
- Problems with scaling RLHF pipelines and **ethical concerns** regarding using human annotator, how they are treated, and the negative effects of filtering out abusive, undesirable LLM responses on their well-being. For more on this, read MIT AI-Risk Initiative’s [Report on OpenAI’s Kenyan Annotators Incident](https://airisk.mit.edu/navigator#/incidents/450).

Let’s work within the framework of RLHF itself however.  A lot of modern learning objectives also make use of the KL divergence penalty to ensure that the post-trained model does not stray too far from the original SFT model and risk losing its learned capabilities. Let’s assume we can define a reward model that accurately rewards our ideal behavior…Even so, it may not be the case that our learning objective will enable the desired behavior change in our model. There seems to be an **inherent tension** between the goal of RL and the way we have designed our objectives. The reward model and objective wants the policy to remove the undesirable behaviors while the KL-regularization penalty wants it to stay as close to the reference behavior as possible. In this blog, we want to study the theoretical bounds KL-constraint imposes on behavior change. More specifically, *how much can an optimal policy reduce the undesirable behavior given that it is constrained by KL-regularization?* 

---

## Toy Model

Let’s start with a very simplified model to build our intuition for the more involved proofs to come. Suppose the situation is as follows.  A user asks a question to our policy: “Can you tell me whether my answer is correct: $4\times5=25$?”

Given the prompt, our toy policy $\pi$ gives one of two responses: agree with the user’s wrong claim (e.g. “Yes, that’s right”) or correct the user (”I think it’s supposed to be $20$”). For a prompt $x,$ $\pi$ outputs the probability $p$ that the response agrees with the false claim—and the reponses get sampled stochastically based on that probability.

$A:= \text{endorse the false claim}$
$C:= \text{correct the user}$
$p:=\text{prob of fake endorsement under reference policy }\pi_0$
$q:=\text{prob of A under new policy }\pi$

Let’s introduce rewards for exhibiting behavior $A$ and $C$ now. Assign $r(A)=0, r(C)>0$. We define a new quantity $\Delta=r(C)-r(A)$, denoting the **reward advantage ****our policy gets for correcting the user over endorsing the false claim.

Then, the expected reward for the new policy $\pi$ is the same as the expected value under its output $q$ :

$$
\mathbb{E}_{\pi}[r]=q\cdot0\,+(1-q)\cdot\Delta = (1-q)\Delta
$$

We now define an objective $J(q)$ which tells us how *good* our policy’s behavior is. Here, we also add a KL-regularization penalty to define our constrained optimization problem.  

$$
J(q) =\mathbb{E}_q[r]-\beta D_{KL}(q,p)
$$

Since our reward model incentivizes correcting the user’s false claims over agreeing with them, maximizing the above objective would ideally **correct** the sycophancy displayed by our reference model. But, *how much* would the behavior change under the KL constraints? To do this, we find an optimal probability $q^*\in(0,1)$ such that it maximizes $J$ and then analyze that.

Let’s start by simplifying the objective further:

$$
J(q) = (1-q)\Delta-\beta(\mathbb{D}_{KL}(\pi||\pi_0))\\\implies J(q) = (1-q)\Delta-\beta\sum_y \pi(y|x)\log\left(\dfrac{\pi(y|x)}{\pi_0(y|x)}\right)\\\therefore J(q)= (1-q)\Delta-\beta\left[q\log\dfrac{q}{p}+(1-q)\log\dfrac{1-q}{1-p}\right]
$$

Clearly the two terms are exerting opposite pressures on our objective. Assume move $q$ away from $p$ — to reduce how often our new policy agrees with false claims. Note that the first (reward) term goes higher as $q\rightarrow 0$ and **pulls $J(q)$ up**. Meanwhile,  the KL divergence penalty also increases as $q$ goes further from $p$ and it **pulls $J(q)$ down.** 

#### Differentiating the reward

We find the optimal policy $q^*$ by finding the stationary points of $J$.

$$
\frac{d}{dq}J(q)= \frac{d}{dq}[(1-q)\Delta]-\beta\left[\frac{d}{dq}[q\log\dfrac{q}{p}]+\frac{d}{dq}[(1-q)\log\dfrac{1-q}{1-p}]\right]
$$

Let’s evaluate the derivatives on the RHS term by term. Starting with the reward term:

$$
\frac{d}{dq}[(1-q)\Delta]=-\Delta
$$

Now, the first term of the KL-divergence penalty:

$$
\frac{d}{dq}[q\log\dfrac{q}{p}]=(\frac{d}{dq}q\cdot\log\dfrac{q}{p})+(q\cdot\frac{d}{dq}[\log\dfrac{q}{p}])\\\therefore \frac{d}{dq}[q\log\dfrac{q}{p}]=\log\dfrac{q}{p}+1\tag{1}
$$

Finally, for the second term of the KL-divergence, applying the product-rule again, we get:

$$
\frac{d}{dq}[(1-q)\log\dfrac{1-q}{1-p}]=-\log\dfrac{1-q}{1-p}-1\tag{2}
$$

Adding equations $(1)$ and $(2)$, gives us the derivative of the KL-divergence:

$$
\therefore \frac{d}{dq}\mathbb{D}_{KL}(\pi||\pi_0)=\log\frac{q}{p}-\log\frac{1-q}{1-p}\\=\log(\frac{q}{1-q})-\log(\frac{p}{1-p})
$$

But notice that both these terms are just the log-odds/**logits** of $q,p$! 

$$
\boxed{\therefore \frac{d}{dq}\mathbb{D}_{KL}(\pi||\pi_0)=\text{logit}(q)-\text{logit}(p)}
$$

This is a neat way of looking at what the KL divergence does. If $q<p$, then the derivative would be **negative,** encapsulating the idea that increasing $q$ would reduce the KL-divergence between $\pi$ and $\pi_0$ (since doing so would bring their probability distributions closer). And vice-versa for $q>p!$ In even simpler terms, the **farther** we move $q$ from $p$, the more the KL-divergence increases. 

Combining the derivatives of the reward and the KL-divergence gives us $J'$:

$$
\boxed{J'(q)=-\Delta-\beta[\text{logit}(q)-\text{logit}(p)]}\tag{3}
$$

#### Finding the optimum

By now I’m sure you’re getting a sense of where this may be heading. It seems like our optimum policy will depend upon where we started ($p$), how big our reward advantage is ($\Delta$), and the strength of the regularization ($\beta$). 

Let’s work out the specifics. Equating to $(3)$ to zero and rearranging:

$$
\text{logit}(q^*)=\text{logit}(p)-\dfrac{\Delta}{\beta}
$$

Exponentiating both sides:

$$
\frac{q^*}{1-q^*}=\frac{p}{1-p}\cdot e^{-\Delta/\beta}\\

\text{odds of $q^*$} =\text{odds of p} \times \text{reward vs }D_{KL} \text{ trade-off}
$$

This equation is already quite informative. It tells us that given an objective constrained by the KL-regularization penalty, how much we would be able to correct the ‘bad’ behavior $A$ depends on two things:

1. <u>How high/low was our initial value.</u> The optimal policy $q^*$ corrects the odds of the reference policy $p$ of endorsing a false claim. 

2. <u>How much we correct depends exactly on the reward-penalty tradeoff.</u> If the reward to correct the reference’s behavior is much stronger compared to the penalty of straying far away from the reference, the optimal policy will result in much more behavioral change. This is captured by the $\dfrac{\Delta}{\beta}$ term. 

To find a closed-form solution for the optimal probability $q^*$, we use the sigmoid $\sigma$, which is the inverse of the logit.

$$
\begin{aligned}
q^*
&= \sigma\left(\operatorname{logit}(p)-\frac{\Delta}{\beta}\right) \\[4pt]
&= \frac{1}{1+\exp\left(-\operatorname{logit}(p)+\frac{\Delta}{\beta}\right)} \\[4pt]
&= \frac{1}{1+\exp\left(-\log\frac{p}{1-p}\right)e^{\Delta/\beta}} \\[4pt]
&= \frac{1}{1+\frac{1-p}{p}e^{\Delta/\beta}}\\&\boxed{\therefore q^*=\frac{p}{p+(1-p)e^{\Delta/\beta}}}\end{aligned}
$$

Clearly, since $p\in(0,1)$ and both $\Delta,\beta>0$, we have $p<p+(1-p)e^{\Delta/\beta}$. And so, $q^*\in(0,1)$ and our optimal solution is a **valid probability.** 

#### How much pressure gets us under a threshold?

The above derivations now allow us to answer more interesting questions about behavior alignment within our toy model. Let’s say we have found that our pre-trained model agrees with false claims with a **really high probabilit**y $p=0.9.$ In our post-training phase, we now want to bring this under a threshold $\eta=0.01$ that we deem as the *target behavior*. What reward and $\beta$ values would get us there?

First, let’s measure the extra KL (coding) cost imposing this new threshold on our policy would incur. Think of this as quantifying in *nats* (a natural-log version of bits) how much divergence our model will incur. If you’re not used to thinking about entropy from an information encoding lens, watch [this 3Blue1Brown video](https://www.youtube.com/watch?v=GlYgs6v2YfU). In practice, we may want to keep the KL-cost low since too much model-drift may risk losing capabilities we picked up in the pre-training and SFT phases.

$$
D_{\mathrm{KL}}(\pi\|\pi_0)= 0.01\ln\frac{0.01}{0.90} + 0.99\ln\frac{0.99}{0.10}\\[4pt]\approx -0.045 + 2.270\\[4pt]= \boxed{2.225\text{ nats}}
$$

Now suppose we decide that this cost trade-off is okay to make. We can now calculate the reward advantage $\Delta$ and the KL-coefficient $\beta$ needed to reach that threshold.

$$
q^*\le\eta\\\implies \operatorname{logit}(p)-\frac{\Delta}{\beta}\le \operatorname{logit}(\eta)\\\implies \boxed{\frac{\Delta}{\beta}\ge\operatorname{logit}(p)-\operatorname{logit}(\eta)}
$$

And for the opening example, where $p=0.9$ and we want $\eta=0.01$, we get

$$
\frac{\Delta}{\beta}\ge \log 9-\log\frac{0.01}{0.99}=\log 891\approx \boxed{6.792}
$$

That is, the reward advantage received from correcting a user’s claim over agreeing with them has to be $6.79$ times more than the KL-coefficient.

The entire derivation above answers two questions that are easy to conflate. First, how far must the policy move from the reference to meet our target behavioral threshold? This can be calculated by the KL-cost of fixing behavior. Second, what reward pressure would make the ideal policy choose that change? This is answered by calculating $\frac{\Delta}{\beta}$, which for our model told us the reward must be $6.7$ times higher.

---

## Generalizing to LLMs

#### **Quick Note**

In our toy model, we considered a single prompt and just two possible response: agreeing with the user or correcting them. Clearly, that example trivializes a lot of details that may need to be thought out much better when dealing with LLMs. Let’s continue on the thread of sycophancy in models to illustrate this. LLMs produce a large variety of responses, which complicates the process of identifying which responses exhibit a behavior like sycophancy. 

Suppose we want to reduce sycophancy through post-training. To do this correctly, we must ensure that we avoid the limitations of RLHF—such as biased annotator preferences—that may have introduced this behavior in the model. Several practical questions arise before we can even begin post-training:

- **How do we define the behavior?** We want to be able to come up with a definition or rubric that distinguishes between appropriate agreement, unwarranted agreement, being overly critical, etc. This is very similar to specifying an *ideal functionality* in security: we first need to define the behavior we want before we can assess it.
- **Which prompts do we test on?** Users may came to LLMs with a wide range from factual mistakes to asking for opinions to emotional advice. Our model’s behavior may vary across these different contexts. Designing an evaluation set which captures a useful range of these contexts is a non-trivial task.
- **How do we judge responses and construct a reward?** Human annotators, classifiers, and LLMs-as-a-judge can all make mistakes in assessing the correct behavior despite having a clear rubric given to them. This effect is amplified when responses are out of their domain of knowledge or more subjective in nature. Any reward model trained on these errors may inculcate them in the model.

Researchers have designed creative experiments to tackle aspects of these problems. In [Understanding Sycophancy in LMs](https://arxiv.org/html/2310.13548), Anthropic researchers created an evaluation set that tests sycophancy through different situations that may induce the behavior.  For example, they tested whether LLMs changed their stance when the user mentioned if they liked or disliked an opinion; or they pressure-tested assistant’s views by challenging them and measured how many correct answers became incorrect after user push-back.

This blog tries the explore the ideal policy more theoretically though. So we’re going to make simplifying assumptions to the practical problems listed above. These are all highlighted in the section below.

#### **The Theoretical Question**

Assume that for a specific prompt, we have <u>specified</u> a set of responses that characterize the undesirable behavior. Also assume that we have <u>trained</u> a reward model that incentivizes the behavior we want and penalizes undesirable behavior. The question we now explore is: *how much can an optimal policy reduce that undesirable behavior, given that it is constrained by KL-regularization?* 

Some notational conventions we will follow in this blog:

$x:=$ $\text{the user prompt}$
$y:=$ $\text{the output generated by the LLM}$ 
$H_x:= \{y: y\text{ is an undesirable  response to prompt $x$}\}$ 
$\rho(x):= \text{the training prompt distribution, i.e. how much we want each prompt to contribute to the training}$$\pi_0(y|x):=$ $\text{the fixed, reference policy; given prompt $x$, it outputs the probability of a response $y$ being generated}$
→ $\pi_{\theta}(y|x):=$ $\text{the new policy with weights $\theta$}$
→ $r(x,y):= \text{the fixed reward earnt by a response to a prompt}$ 

#### Connecting complete responses to tokens

Our policy above describes a probability over responses $y$ for a prompt $x$. To reduce the undesirable behavior in $\pi_0$ , we want the new policy $\pi$ to have the following property: 

$$
\sum_{y\in H_x} \pi_\theta(y|x)<\sum_{y\in H_x} \pi_0(y|x)
$$

That is, the probability that a undesirable response is generated by our new policy must be lower than the reference. Till now, we have been working on a response level. However, LLMs don’t output complete responses; they work on a token-level. How do we translate between responses and tokens? Responses can easily be decomposed into a chain of token-level outputs that form that response. 

Let’s model LLM text-generation as an MDP. 

- $s_t=(x,y_1,y_2,...y_{t-1})$, the current state is the prompt and the text generated by the LLM uptil now.
- $a_t=y_t$, the next token prediction given the current state
- $y = (x,y_1,y_2,...,y_n)$, the complete response can be thought of as a trajectory of actions

The probability of a response being generated by our policy is thus the product of the probability that at each time step, our LLM generated the $t$-th token in our response.

$$
\pi_\theta(y|x) = \prod_{t=1}^n\pi_\theta(y_t|x,y_{<t})
$$

<u>A caveat:</u> Changing a model’s shared weights affects the probabilities of multiple tokens and responses at once. Even if we find an optimal policy $\pi^*$, that reduces the undesirable behavior, there is no guarantee that LLM weights could represent that or that our learning algorithm could find it. More on this later.

#### Defining the Parameterized Objective

We set up our toy model such that the policy directly outputs the probability of agreeing to the false claim and thus we could define our objective $J(q)$ in terms of that. We now want to move to the more general case where we have a set of weights and the model outputs a distribution of tokens which come together to form a response. Note that this process is stochastic—an LLM may reply to the same prompt with different responses each time you sample. Therefore, we define the new parameterized objective $J(\theta)$ as follows:

$$
J(\theta)=
\mathbb{E}_{x\sim\rho} \left[ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)} \left[r(x,y)\right] - \beta D_{\mathrm{KL}} \left(\pi_\theta(\cdot\mid x) \,\|\, \pi_0(\cdot\mid x) \right)\right]
$$

The definition above looks daunting at first. A good way to break down such nested expectations is by flipping your perspective from trying to understand their ‘meaning’ to thinking about how we’d **actually calculate** them in practice. Defining objectives using expectations makes them really easy to calculate.

The process we’d follow is to calculate the sample average by generating multiple responses for a prompt, and then use Law of Large Numbers to argue that it's a good approximation of the expectation, giving us a close enough estimate of $J(\theta)$.

A more granular break down:

1) Sample a prompt $x$ from the training distribution $\rho$.
2) For that prompt, generate multiple LLM responses $y$.
3) For each response, calculate the $r(x,y)-\beta D_{\mathrm{KL}}$  term.
4) Average over all sample responses $y$. This gives us an estimate of *how good the model behavior is on a single prompt $x$.*
5) Repeating for more samples of $x$ and then averaging it out tells us *how good the model behavior is across our evaluation set.* This is exactly what $J(\theta)$ is trying to model!

#### **Deriving the gradient-update rule**

For a fixed prompt $x$, reward model $r$, and reference $\pi_0$, we get:

$$
\begin{aligned}
J_x(\theta)

&=

\mathbb{E}_{y\sim\pi_\theta}

\left[

r(y)- 
\beta\log\frac{\pi_\theta(y)}{\pi_0(y)}

\right] \\

&=

\sum_y \pi_\theta(y)r(y)-
\beta\sum_y

\pi_\theta(y)

\log\frac{\pi_\theta(y)}{\pi_0(y)}

\end{aligned}
$$

We now derive the gradient-update rule for this generalized objective as follows:

$$
\nabla_\theta\,J_x(\theta) =
\nabla_\theta
\sum_y \pi_\theta(y)r(y)-
\beta\cdot\nabla_\theta\left(\sum_y\pi_\theta(y)\log\frac{\pi_\theta(y)}{\pi_0(y)}\right)
$$

Let’s evaluate the RHS term-by-term. First, the reward term:

$$
\nabla_\theta
\sum_y \pi_\theta(y)r(y)=
\sum_y r(y)\cdot \nabla_\theta\pi_\theta(y)\tag{4}
$$

Think about where we want to take this derivation. Ultimately—and this is a common trick used across policy gradient methods—we’d like to re-express the gradient as an expectation of something (for the same reasons mentioned before).

Recall, the log-derivative trick: $\nabla_\theta \pi_\theta(y)=\pi_\theta(y)\cdot\nabla_\theta\log(\pi_\theta(y))$. Substituing this in  $(4)$:

$$
\implies \sum_y r(y)\cdot \nabla_\theta\pi_\theta(y)=
\sum_y
r(y)\cdot\pi_\theta(y)
\nabla_\theta\log\pi_\theta(y) \\
\boxed{\therefore \nabla_\theta
\sum_y \pi_\theta(y)r(y)=
\mathbb{E}_{y\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(y)\,r(y)
\right]}\tag{*}
$$

Now, let’s unpack at the KL term:

$$
\beta\cdot\nabla_\theta
\sum_y
\pi_\theta(y)
\log\frac{\pi_\theta(y)}{\pi_0(y)}
=
\beta\sum_y
\nabla_\theta
\left[
\pi_\theta(y)\log\pi_\theta(y)
-
\pi_\theta(y)\log\pi_0(y)
\right]
$$

Applying the product rule:

$$
=
\beta\sum_y
\nabla_\theta\pi_\theta(y)
\left[
\log\pi_\theta(y)+1
-
log\pi_0(y)
\right]\\
=
\beta\sum_y
\nabla_\theta\pi_\theta(y)
\left[
\log\frac{\pi_\theta(y)}{\pi_0(y)}
+1
\right]
$$

Once again using the log-identity rule:

$$
=
\beta\sum_y
\pi_\theta(y)
\nabla_\theta\log\pi_\theta(y)
\left[
\log\frac{\pi_\theta(y)}{\pi_0(y)}
+1
\right]\\
\boxed{\therefore \beta\,\nabla_\theta
\sum_y
\pi_\theta(y)
\log\frac{\pi_\theta(y)}{\pi_0(y)} = \beta\,\mathbb{E}_{y\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(y)
\left(
\log\frac{\pi_\theta(y)}{\pi_0(y)}
+1
\right)
\right]}\tag{**}
$$

Combining the two terms, we finally get the gradient-update rule

$$
\nabla_\theta J_x(\theta)
=
\mathbb{E}_{y\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(y)
\left(
r(y)
-
\beta\log\frac{\pi_\theta(y)}{\pi_0(y)}
-
\beta
\right)
\right]
$$

 Since the expected value of the gradient of the log-probabilities is zero, we can remove the constant $\beta$ term.

$$
\boxed{\nabla_\theta J_x(\theta)
=
\mathbb{E}_{y\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(y)
\left(
r(y)
-
\beta\log\frac{\pi_\theta(y)}{\pi_0(y)}
\right)
\right]}

$$

Policy-gradient algorithms can use this to hill-climb our objective and help instill the aforementioned *‘good’ behavior* in our new policy $\pi_\theta$.

---

## Finding the Optimal KL-Constrained Policy

We now have all the tools to answer our research question. *How much can the optimal policy reduce the behavior that we want to change?* This section will follow the proof outlined in Appendix A.1 of [Rafailov et.al’s paper](https://arxiv.org/pdf/2305.18290) on Direct-Preference Optimization.

Here’s a quick sketch of the proof so it will be easier to follow. First, we will fix a prompt and try to find the policy that maximizes our objective **over all possible distributions of responses** for that prompt. To do this, borrowing from our intuition from the toy model, we will define a candidate $\pi^*$. And then, weprove that it is optimal.

Note how this proof digresses from our toy-model direction of finding the stationary points of $J(q)$. Why don’t we do the same here by setting $\nabla_\theta J(\theta)=0?$ Solving for stationary points may not lead us to the **global optimum** in parametrized spaces**.** Knowing which points are global maximas and not local maximas or saddle points would require knowledge about the landscape of $J$ and the weights. Instead, we go the cleaner route by optimizing directly for the ideal policy: *what is the $\pi^*$ that maximizes the objective compared to **all possible distributions** over the final responses $y$?*

#### Start with the problem for one prompt

Fix a prompt $x$. Let’s simplify the notation: write $\pi(y)$ for $\pi(y|x)$ and do the same for $\pi_0(y|x)$ and $r(x,y)$. 

We further assume:

- <u>The set of all complete responses $y$ is finite.</u> This follows form the fact that our token vocabulary is finite and our maximum response length is also finite.
- $\pi_0$ <u>has full support.</u> This assumption is made for the well-definedness of the KL-divergence.
- <u>Reward model $r$ has been trained and fixed; and $\beta>0$.</u>

Recall, our objective $J(\theta)$. We express reparameterize it again: this time in terms $\pi$, a policy giving probabilities over full responses:

$J(\pi)=\mathbb{E}_{y\sim\pi}[r(y)-\beta\mathbb{D}_{KL}(\pi||\pi_0)] = \sum_{y}\pi(y)r(y) - \beta \sum_{y} \pi(y) \log \frac{\pi(y)}{\pi_0(y)}$

**Step 1: Build a candidate distribution**

Remember the optimal probabilities in our toy model:

$$
q^*=\dfrac{p}{p+(1-p)e^{\Delta/\beta}} \quad \quad\quad\quad1-q^*=\dfrac{(1-p)e^{\Delta/\beta}}{p+(1-p)e^{\Delta/\beta}}
$$

Spot any pattern? Both the numerators are weighing the reference probability with a term that measures the reward-penalty ratio, or **pressure** as we described it before. The denominators are simply normalizing so that we end up with values between $0$ and $1$.

 Thus, we can take inspiration from the toy model to construct a good candidate for our optimal policy. Define the weighing function $w$ and normalization constant $Z$:

$$
w(y)=\pi_0(y)e^\frac{r(y)}{\beta}\quad\quad\quad\quad Z=\sum_y \pi_0(y)e^\frac{r(y)}{\beta}=\mathbb{E}_{y\sim\pi_0}[e^{\frac{r(y)}{\beta}}]
$$

And let our candidate policy $\pi^*$ be:

$$
\pi^*(y)=\dfrac{\pi_0 e^{\frac{r(y)}{\beta}}}{Z}=\dfrac{w(y)}{Z}\tag{7}
$$

For each response $y$, this takes gives us a valid probability with $(0,1)$ and thus this is a valid distribution.

**Step 2: Express the reward through our candidate**

Divide the candidate with the reference and take the logarithm on both sides:

$$
\frac{\pi^*(y)}{\pi_0(y)}
=
\frac{
    e^{\frac{r(y)}{\beta}}
}{
    Z
}, \quad \log\frac{\pi^*(y)}{\pi_0(y)}
=
\frac{r(y)}{\beta}
-
\log Z
$$

Rearranging, we get an equation for the reward written in terms of our candidate $\pi^*$ 

$$
r(y) = \beta\log\frac{\pi^*(y)}{\pi_0(y)}+\beta\log Z \tag{5}
$$

In the next step, we will substitute this into the objective.

**Step 3: Rewrite the objective**

$$
J(\pi)=\mathbb{E}_{y\sim\pi}[r(y)-\beta\mathbb{D}_{KL}(\pi||\pi_0)]\\

= \mathbb{E}_{y\sim\pi}[r(y)]-\beta\mathbb{E}_{y\sim\pi}[\mathbb{D}_{KL}(\pi||\pi_0)]\tag{6}
$$

Let’s see what the expectation of the reward is. Putting $(5)$  into our equation:

$$
\mathbb{E}_{y\sim\pi}[r(y)] =\mathbb{E}_{y\sim\pi}[\beta\log\frac{\pi^*(y)}{\pi_0(y)}+\beta\log Z] \\

= \mathbb{E}_{y\sim\pi}[\beta\log\frac{\pi^*(y)}{\pi_0(y)}]+\mathbb{E}_{y\sim\pi}[\beta\log Z]\\
$$

But $Z$ and $\beta$ are just constants for a fixed prompt, so $\beta\log Z$ is constant: 

$$

= \beta\log Z+\mathbb{E}_{y\sim\pi}[\beta\log\frac{\pi^*(y)}{\pi_0(y)}]
$$

Plugging this back into the equation $(6)$ for $J(\pi)$:

$$
J(\pi)= \beta\log Z+\mathbb{E}_{y\sim\pi}[\beta\log\frac{\pi^*(y)}{\pi_0(y)}] -\beta\mathbb{E}_{y\sim\pi}[{D}_{KL}(\pi||\pi_0)]\\

= \beta\log Z + \beta\mathbb{E}_{y\sim\pi}[\log\frac{\pi^*(y)}{\pi_0(y)}-\log\frac{\pi(y)}{\pi_0(y)}]\\

= \beta\log Z +\beta\mathbb{E}_{y\sim\pi}[-\log\frac{\pi(y)}{\pi^*(y)}]\\

\boxed{\therefore J(\pi)= \beta\log Z -\beta\mathbb{E}_{y\sim\pi}[\mathbb{D}_{KL}(\pi||\pi^*)]}
$$

Since $\beta\log Z$ is a constant, maximizing our objective $J$ reduces to minimizing the second term. And since the KL-divergence is always non-negative, it is minimized when ${D}_{KL}=0\iff\pi=\pi^*$. 

Therefore, our candidate $\pi^*$ is the ideal policy that **maximizes $J(\pi)$** over all responses to a fixed prompt $x$. 

#### Going From One Reply to a Behavior

The previous section showed how the optimal policy acts on  each reponse to a fixed prompt. In this section, we will try to see how the optimal policy shifts the behavior of our model on a whole. 

Recall that $H$ refers to the set of responses that we deem as undesirable behavior — in our case this behavior was sycophancy.  Let $p=\pi_0(H)$ denote our reference policy’s probability of exhibiting this behavior. Let $q^*=\pi^*(H)$, probability of our new policy exhibiting $H$. Then,

$$
p=\sum_{y\in H}\pi_0(y) \quad\quad\quad\quad q^*=\sum_{y\in H}\pi^*(y)
$$

We want study how $q^*$(the new probability of exhibiting sycophancy) relates to $p$ (the reference probability).

Substitute the optimal policy equation $(7)$ that we derived:

$$
q^*=\sum_{y\in H}\pi^*(y)=\frac{1}{Z}\sum_{y\in H}\pi_0(y)e^{\frac{r(y)}{\beta}}
$$

$$
1-q^*=\sum_{y\in H^c}\pi^*(y)=\frac{1}{Z}\sum_{y\in H^c}\pi_0(y)e^{\frac{r(y)}{\beta}}
$$

But notice that we can substitute the within-group probabilities $\pi_0(y)=p\,\pi_0(y|H)$ for $y\in H$

$$
q^*=\frac{1}{Z}\sum_{y\in H}\pi_0(y)e^{\frac{r(y)}{\beta}}=\frac{1}{Z}\sum_{y\in H}p\,\pi_0(y|H)e^{\frac{r(y)}{\beta}}\\

= \frac{p}{Z}\sum_{y\in H}\pi_0(y|H)e^{\frac{r(y)}{\beta}}
$$

The term inside the summation is precisely the conditional expectation of the exponential-reward within group $H$!

$$
\boxed{\therefore q^*=\dfrac{p}{z}\mathbb{E}_{y\sim \pi_0}[e^{\frac{r(y)}{\beta}}|H]}
$$

As our intuition would suggest, the optimal policy shifts the probability of exhibiting behavior $H$ by the *average exponential reward weight* for responses in $H$! 

And similar line of reasoning for $H^c$ gives us:

$$
\boxed{1-q^*=\dfrac{1-p}{z}\mathbb{E}_{y\sim \pi_0}[e^{\frac{r(y)}{\beta}}|H^c]}
$$

Now, we conclude by calculating the **odds of $\pi$ behaving sycophantically:**

$$
\dfrac{q^*}{1-q^*}=\dfrac{p}{1-p}\dfrac{\mathbb{E}_{y\sim \pi_0}[e^{\frac{r(y)}{\beta}}|H]}{\mathbb{E}_{y\sim \pi_0}[e^{\frac{r(y)}{\beta}}|H^c]}
$$

The odds of behaving like $H$ are equal to the reference odds multiplied by the ratio between the average reward weights for responses in $H:H^c$. Therefore, the new odds shift up if our $r(x,y)$ rewards responses in $H$ **more (on average)** than responses in $H^c$.

#### Optimal Policy for All Possible Prompts

Till now, we found the optimal policy for a fixed prompt $x$. But if we sample the prompts identically from the training distribution $\rho$, applying the ideal per-prompt policy $\pi^*(y|x)$ gives us the optimal responses for each sampled prompt. Then, by linearity of expectation, averaging over all the per-prompt policies would give us the distribution that maximizes the objective on average for any prompt $x\sim\rho.$ 

#### Practical Problems with Finding $\pi^*$

Our entire derivation of the optimal policy bypassed working with the model weights $\theta$ and directly found the optimal distribution $\pi^*$by comparing it with other distributions over complete reponses. Taking this approach, invariably leads to questions about whether such an optimal policy can actually be attained in practice. This boils down to three main questions:

1. <u>Representability:</u> Is there a set of weights $\theta^*$ that induce a token-level distribution such that it leads to the optimal response distribution $\pi^*$?

2. <u>Optimization:</u> Can a learning algorithm like PPO or DPO find those weights $\theta^*$?

3. <u>Approximatability:</u> If we cannot replicate the exact distribution, how close can we get to the ideal policy? Is there a $\epsilon$ such that, $\mathbb{D}_{KL}(\pi_\theta|\pi^*)<\epsilon$?

These make for exciting research directions that bridge the gap between the theory and practice!

---

That’s it from my side today! Hopefully I did this beautiful proof justice and you found it easy to follow along. If you’re someone who enjoys the Math just as much as me or if any of these research directions excites you, please do reach out to me on X. 

Until next time,
-Vatsl.
*There is much to learn and much more to see*