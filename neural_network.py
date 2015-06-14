#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

class NeuralNetwork:
    def __init__(self, in_size, hidden_size, out_size):
        self.w = np.random.random([hidden_size, in_size + 1])
        self.v = np.random.random([out_size, hidden_size + 1])
        self.eta = 0.1
        self.alpha = 0.1

    def forward(self, x):
        y = sigmoid(np.dot(self.w, x))
        y = np.append(y, 1)
        z = sigmoid(np.dot(self.v, y))
        return y, z

    def bp(self, x, t, y, z):
        delta_o = z * (1 - z) * (t - z)
        delta_h = y * (1 - y) * np.dot(delta_o, self.v)
        d_v = self.eta * delta_o * y
        self.v = self.v + d_v
        d_w = self.eta * np.outer(delta_h[:-1], x.T)
        self.w = self.w + d_w


    def calculate_error(self, T, Z):
        return (1/T.size) * sum((t-z)**2 for t, z in zip(T, Z))

    def _fit(self, x, t):
        x = np.append(x, 1)
        y, z = self.forward(x)
        self.bp(x, t, y, z)
        return z

    def fit(self, X, T, eps=10e-3):
        error = sys.maxsize
        while error > eps:
            Z = np.array([self._fit(x, t) for x, t in zip(X, T)])
            error = self.calculate_error(T, Z)
            print(error)


    def predict(self, X):
        for x in X:
            x = np.append(x, 1)
            _, z = self.forward(x)
            print(z)

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    T = np.array([1,0,0,1])
    clf = NeuralNetwork(X.shape[1], 20, 1)
    clf.fit(X, T)
    clf.predict(np.array([[0,0.1]]))

