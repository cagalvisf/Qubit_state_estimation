import numpy as np

def linear_estimation(T, p):
    ## Estimation by linear inversion

    s = np.array(np.matmul(np.linalg.inv(T),p), ndmin=0)
    s = np.reshape(s,4)
    
    s1 = s[1].real
    s2 = s[2].real
    s3 = s[3].real

    s = [s1, s2, s3]
    
    return np.array(s)

    ## Discrete Maximum likelihood estimator
def disc_ML_estimation(T, p):

    T = np.array(T)
    se = np.array([1.0,0.0,0.0,0.0])
    nint = 10000

    for _ in range(1,nint):
        pe = np.dot(T,se)

        re = np.dot(np.transpose(T),(p/pe))

        ge = re[1]**2 + re[2]**2 + re[3]**2 - re[0]**2
        se[1] = (2*re[1]-se[1]*ge)/(2*re[0]+ge)
        se[2] = (2*re[2]-se[2]*ge)/(2*re[0]+ge)
        se[3] = (2*re[3]-se[3]*ge)/(2*re[0]+ge)
    
    s = np.reshape(se,4)
        
    s1 = s[1].real
    s2 = s[2].real
    s3 = s[3].real

    s = [s1, s2, s3]
    
    return np.array(s)