#!/usr/bin/env python3
"""Agent-independent feasibility ceiling for G0-A (PREREG deviation D1).

Simulates reference algorithms on the exact gate task distribution
(8-arm Bernoulli, p ~ U(0,1), 200-pull lifetime) and reports the
final-quarter best-arm rate. Thompson sampling is the reference for the
relative gate criterion; UCB1 is a sanity floor.
"""
import numpy as np

rng = np.random.default_rng(0)
N, K, T = 4000, 8, 200
q = T - T // 4


def thompson():
    hits = 0
    for _ in range(N):
        p = rng.uniform(0, 1, K)
        best = p.argmax()
        a_, b_ = np.ones(K), np.ones(K)
        for t in range(T):
            arm = rng.beta(a_, b_).argmax()
            r = rng.random() < p[arm]
            a_[arm] += r
            b_[arm] += 1 - r
            if t >= q and arm == best:
                hits += 1
    return hits / (N * (T - q))


def ucb1():
    hits = 0
    for _ in range(N):
        p = rng.uniform(0, 1, K)
        best = p.argmax()
        n, s = np.zeros(K), np.zeros(K)
        for t in range(T):
            arm = t % K if t < K else (s / n + np.sqrt(2 * np.log(t + 1) / n)).argmax()
            r = rng.random() < p[arm]
            n[arm] += 1
            s[arm] += r
            if t >= q and arm == best:
                hits += 1
    return hits / (N * (T - q))


if __name__ == "__main__":
    print("Thompson final-quarter best-arm rate:", round(thompson(), 3))
    print("UCB1     final-quarter best-arm rate:", round(ucb1(), 3))
