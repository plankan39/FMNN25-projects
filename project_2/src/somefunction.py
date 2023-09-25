import numpy as np
def newton(f,point,epsilon,max_iter):
    i = 0
    next_point = point - np.dot(np.linalg.inv(HESSIAN) ,GRADIENT)
    while  np.linalg.norm(point - next_point) >  epsilon and i < max_iter:
        
        i += 1
        next_point = point - np.dot(HESSIAN,GRADIENT)
    return next_point


<<<<<<< HEAD
def calculate_gradient(f, p, epsilon=0.01):
    p = np.array(p)
    gradient = np.zeros_like(p)
    
    for i in range(len(p)):
        p_shifted_front = p.copy()
        p_shifted_back = p.copy()
        
        print("Before addition:", p_shifted_front)
        p_shifted_front[i] += epsilon
        print("After addition:", p_shifted_front)
        
        p_shifted_back[i] -= epsilon
        
        gradient[i] = (f(*p_shifted_front) - f(*p_shifted_back)) / (2 * epsilon)
        
    return gradient

def func(x,y):
    return x*x + y*x

p = np.array([1,2], dtype=float)
print(calculate_gradient(func, p, epsilon=0.1))

