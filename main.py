import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint

r=0.5
c=0.5
L=0.2
A=np.array([[0,1], [-1 / (L * c), -r / L]])
B=np.array([[0],[1/L]])
q = np.array([[1, 0], [0, 1]])
R = 1
P = scipy.linalg.solve_continuous_are(A, B, q, r)
R_inv = np.array([1])
BT = np.transpose(B)
K = R_inv @ BT @ P

# x1,x2
def model(z,t,modified_u):
    x=np.array([[z[0]],[z[1]]])
    if modified_u:
        u=-K@x
        dx = A @ x + B * u[0]
        return np.array([dx[0, 0], dx[1, 0]])
    else:
        dx = A @ x + B * 1
        return np.array([dx[0, 0], dx[1, 0]])

def zadanie1(active):
    if active:
        q=np.array([[1,0],[0,1]])
        R=1
        P = scipy.linalg.solve_continuous_are(A, B, q, r)
        print(P)
        R_inv=np.array([1])
        BT=np.transpose(B)
        K=R_inv@BT@P
        t=np.linspace(0,5,101)
        baseSystem=odeint(model,y0=[0,0],t=t,args=(False,))
        plt.figure('Base system')
        plt.plot(t,baseSystem[:,0],label='x1')
        plt.plot(t, baseSystem[:, 1], label='x2')
        plt.legend()
        modifiedSystem=odeint(model,y0=[0.1,0.1],t=t,args=(True,))
        plt.figure('Modified system')
        plt.plot(t, modifiedSystem[:, 0], label='x1')
        plt.plot(t, modifiedSystem[:, 1], label='x2')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    zadanie1(True)