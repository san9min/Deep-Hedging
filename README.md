# Deep-Hedging
*In 2021 FALL, Financial-Engineering*

## Introduction
 According to Black-Scholes, we can hedge a call option whose pay-off diagram is max(S-K,0) and a put option whose max(K-S,0). We can also hedge a financial derivative which is made of call and put options with their help. But Black-Sholes cannot hedge all financial derivatives. Because they use the way called delta hedging,
They can only predict the financial derivative based on call and put option. We want to solve this problem by using Neural Network and we call this solution Deep Hedging. 
 The idea is Neural Network induces the proper delta. But, can Neural Network find the delta? To answer this, I’ll show followings:
 First, you can hedge an option by using Neural Network without Black-Sholes formula. And then, you can also hedge a financial derivative(option strategy) which is composed of call and put options. Second, I’ll drive a premium, which is the price of an option or a financial derivative. 
 if we success above, we claim Neural Network can find the delta and thus I suggest that with Deep Hedging you can hedge any option strategies that have any pay-off, which is Black-Sholes can’t hedge. Then, the problem is how Neural Network can find the best hedging delta. We think that you can solve this problem with Recurrent Neural Network(RNN). The RNN store all previous data, so you can use this to find the delta, considering the previous all stock prices.
 
 ---
 ## Black-Sholes formula
 
 The first is what the Black-Sholes Option Pricing Formula is. We start with the binomial option pricing formula. They assume that there is no arbitrage, the interest rate is constant and the stock price follows a multiplicative binomial process over discrete periods. The rate of return on the stock over each period can have two possible values: u- 1 with probability q, or d - 1 with probability 1-q. Thus, if the current stock price is S, the stock price at the end of the period will be either uS or dS.

$$
S \quad = \quad 
\begin{cases}
uS \qquad \cdots \quad (1) \\
\\
dS \qquad \cdots \quad (2) 
\end{cases}
$$
 
 Let C be the current value of the call, C_u be its value at the end of the period if the stock price goes to uS, that is, (1) and C_d be its value at the end of the period if the stock price goes to dS, that is, (2). if there is now only one period remaining in the life of the call, we know that the terms of its contract and a rational exercise policy 
imply that Cu = max[0, uS-K] and Cd = max[0, dS –K].

$$
C \quad = \quad 
\begin{cases}
C_u = max[0, uS-K] \qquad \cdots \quad (3) \\
\\
Cd = max[0, dS –K] \qquad \cdots \quad (4)
\end{cases}
$$

Suppose we form a portfolio containing $\Delta$ shares of stock and the dollar 
amount B in riskless bonds. This will cost $\Delta S + B$. At the end of the period, 
the value of this portfolio will be 

$$
\Delta S + B \quad = \quad 
\begin{cases}
\Delta uS + rB \qquad \cdots \quad (5)  \\
\\
\Delta ds + rB \qquad \cdots \quad (6)
\end{cases}
$$



Since we can select ∆ and B in any way we wish, suppose we choose them to 
equate the end-of-period values of the portfolio and the call for each possible 
outcome. This requires that **(3)** must be equal to **(5)** and **(4)** must be equal to **(6)**.

$$
\begin{align}
C_u = \Delta uS + rB \\
\\
C_d =  \Delta dS + rB
\end{align}
$$

Solving these equations, we find

$$ \Delta = \frac{C_u - C_d}{(u-d)S}$$

With ∆ and B chosen in this way, we will call this the hedging portfolio. 
That is, if we have the delta stocks, we hedge the option.
In general, Option pricing formula is

$$
\begin{align}
C = SN(x) - Kr^{-t}N(x-\sigma \sqrt t) \quad \cdots \quad (7) \\
\text{where } x = \frac{log(S/Kr^{-t})}{\sigma \sqrt t} + \frac 1 2 \sigma \sqrt t
\end{align}
$$

Delta, which is the number of stocks we have, is

$$
\Delta \quad =  \quad
\begin{cases}
N(d1) \quad \text{for Call }  \\
N(d1) -1 \quad \text{for Put} \\
\end{cases}
\begin{align}
\quad \text{where } \quad d1 = \frac {ln (S/K) + (r+ \frac 1 2 \sigma^2)t}{\sigma \sqrt t} \quad \text{and} \quad d2 = d1 - \sigma \sqrt t
\end{align}
$$

By using these, the option price for Call is
$$ C_{call} = SN(d_1) - K e^{-rt} N(d2) \quad \cdots \quad (8)$$
and for Put
$$ C_{put} = K e{-rt} N(-d2) -SN(-d_1) \quad \cdots \quad (9)$$

So, if we know stock price, delivery price, remaining periods before its expiration date, 
volatility and interest rate, we can drive a option price. 

The stock price is predicted as
$$ S_{t+\Delta t} = S_t(1+N(r\Delta t, \sigma^2 \Delta t^2))\quad \cdots \quad (10)$$
Note that $\Delta$ is just difference, not hedging delta.

Then, the day cost for hedging is that the product of displacement of delta and stock price at that time. So, the total cost is just sum of the day costs. We call this cost the hedging cost. And it can be represented as

$$
\begin{align}
\Delta_0 S_0 + (\Delta_1 - \Delta_0)S_1 + (\Delta_1 - \Delta_0)S_2 + \quad \cdots \quad +(\Delta_{t-1} - \Delta_{t-2})S_{t-1} +(\Delta_{t} - \Delta_{t-1})S_{t} \\
\text{ or}\qquad \Delta_0(S_0 - S_1) + \Delta_1(S_1- S_2) + \Delta_2(S_2- S_3) + \quad \cdots \quad + \Delta_{t-1}(S_{t-1} - S_t) + \Delta_t S_t
\end{align}
$$

---

## Delta hedging
 Suppose you sell an option and you want to hedge this. Then, you can get help from Black-Sholes. For example, let you sell a call option. All you have to do is just to calculate delta by using **(8)**. Or a put option, **(9)**. you hedge the option if and only If you have the same number of stock as delta. Because we suppose there is no arbitrage, the sum of the cost that you spend while hedging the option and pay-off of the option must be same as the option’s premium. Then, how can we hedge the option strategy that can’t get help from Black-Sholes? This is the reason why we suggest the Deep Hedging.
 
---

## Neural network for delta hedging; *Deep Hedging* 
We want to make a neural network regress to the hedging delta.
Let’s start with option hedging. For simplicity, let you sell a call option and hedge this. And suppose that interest rate, volatility is constant and equal to 0, 0.2 respectively.
Let current stock price S0 be 1.00, delivery price K 1.00, remaining periods before its expiration date 30/365. Then according to **(10)**, we can make scenarios that represent the stock price through time and let this S. Also, we can calculate the option price by **(10)**, or directly **(8)** and **(9)**.
Before using neural network, check the Black-Sholes’ delta is valid. If we hedge the option with Black-Sholes, the sum of the hedging cost, pay-off and premium must be zero. 
The results are:

