#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def forward(A, B, rho, x):
    alpha_1 = rho * B[:, x[0]]
    alpha_2 = np.dot(alpha_1.T, A) * B[:, x[1]]
    alpha_3 = np.dot(alpha_2.T, A) * B[:, x[2]]
    print(alpha_3)

def backward(A, B, rho, x):
    pass

if __name__ == "__main__":
    #A = np.array([[0.15, 0.60, 0.25],
    #              [0.25, 0.15, 0.60],
    #              [0.60, 0.25, 0.15]])
    #B = np.array([[0.5, 0.5],
    #              [0.5, 0.5],
    #              [0.5, 0.5]])
    #rho = np.array([1.0, 0.0, 0.0])
    x = np.array([0.0, 1.0, 0.0])
    A = np.array([[0.1, 0.7, 0.2],
                  [0.2, 0.1, 0.7],
                  [0.7, 0.2, 0.1]])
    B = np.array([[0.9, 0.1],
                  [0.6, 0.4],
                  [0.1, 0.9]])
    rho = np.array([1/3, 1/3, 1/3])

    alpha = forward(A, B, rho, x)
    beta = backward(A, B ,rho, x)

