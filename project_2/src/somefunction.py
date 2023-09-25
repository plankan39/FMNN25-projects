import numpy as np
def newton(f,point,epsilon,max_iter):
    i = 0
    next_point = point - np.dot(HESSIAN,GRADIENT)
    while  np.linalg.norm(point - next_point) >  epsilon and i < max_iter:
        
        i += 1
        next_point = point - np.dot(HESSIAN,GRADIENT)
    return next_point

def calculate_gradient(f, p, epsilon=1e-5):
   
    p = np.array(p)
    
   
    gradient = np.zeros_like(p)
    
    
    for i in range(len(p)):
        
        gradient[i] = (f(p+epsilon) - f(p)) / epsilon
        
    return gradient
 
