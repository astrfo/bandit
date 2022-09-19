import os
from datetime import datetime
import numpy as np
from env import Environment
from policy.e_greedy import e_greedy
from policy.random_policy import random_policy
from policy.ts import TS
from policy.ucb1 import UCB1
from policy.ucb1_tuned import UCB1_tuned
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

class Simulator(object):
    def __init__(self, trial, step, K):
        self.policy = [e_greedy(K), TS(K), UCB1(K), UCB1_tuned(K)]
        self.trial = trial
        self.step = step
        self.K = K
        self.regret = np.zeros(self.step)
        self.make_folder()

    def run(self):
        for policy in self.policy:
            for t in range(self.trial):
                self.env = Environment(self.K)
                self.prob = self.env.prob
                policy.initialize()
                self.regretV = 0.0
                for s in range(self.step):
                    arm = policy.select_arm()
                    reward = self.env.play(arm)
                    policy.update(arm, reward)
                    self.calc_regret(t, s, arm)
            self.save_csv(policy)

    def calc_regret(self, t, s, arm):
        self.regretV += (self.prob.max() - self.prob[arm])
        self.regret[s] += (self.regretV - self.regret[s]) / (t+1)

    def make_folder(self):
        time_now = datetime.now()
        self.results_dir = f'log/{time_now:%Y%m%d%H%M}/'
        os.makedirs(self.results_dir, exist_ok=True)

    def save_csv(self, policy):
        f = open(self.results_dir + 'log.txt', mode='w', encoding='utf-8')
        f.write(f'sim: {self.trial}, step: {self.step}, K: {self.K}\n')
        np.savetxt(self.results_dir + policy.__class__.__name__ + '.csv', self.regret, delimiter=",")
        f.close()