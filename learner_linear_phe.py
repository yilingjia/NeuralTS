import numpy as np
import scipy as sp

class LinearPHE:
    # Brute-force Linear TS with full inverse
    def __init__(self, dim, lamdba=1, nu=1, style='ts'):
        self.dim = dim
        self.U = lamdba * np.eye(dim)
        self.Uinv = 1 / lamdba * np.eye(dim)
        self.nu = nu
        self.jr = np.zeros((dim, ))
        self.mu = np.zeros((dim, ))
        self.lamdba = lamdba
        self.style = style
        self.context_history = []
        self.reward_history = []

    def select(self, context):
        r = np.dot(context, self.mu)
        return np.argmax(r)
        
        # if self.style == 'ts':
        #     theta = np.random.multivariate_normal(self.mu, self.lamdba * self.nu * self.Uinv)
        #     r = np.dot(context, theta)
        #     return np.argmax(r), np.linalg.norm(theta), np.linalg.norm(theta - self.mu), np.mean(r)
        # elif self.style == 'ucb':
        #     sig = np.diag(np.matmul(np.matmul(context, self.Uinv), context.T))
        #     r = np.dot(context, self.mu) + np.sqrt(self.lamdba * self.nu) * sig
        #     return np.argmax(r), np.linalg.norm(self.mu), np.mean(sig), np.mean(r)
        
    def get_jr(self):
        self.jr = np.zeros((self.dim, ))
        for i in range(len(self.context_history)):
            self.jr += (self.reward_history[i] + np.random.normal(0, 0.1)) * self.context_history[i]
        
        return self.jr
        
    def train(self, context, reward):
        self.context_history.append(context)
        self.reward_history.append(reward)
        self.jr = self.get_jr()
        # self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        # fast inverse for symmetric matrix
        zz , _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)
        return 0