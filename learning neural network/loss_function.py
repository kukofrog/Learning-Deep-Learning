def mean_squared_error(y, t):      #mean squared error
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):      #cross entropy error
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))     #Adds delta. If you put 0, it becomes minus infinity.
