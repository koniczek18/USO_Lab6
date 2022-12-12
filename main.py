import numpy as np
import scipy.interpolate
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
K = R_inv @ B.T @ P

# x1,x2, J
def model(z,t,modified_u):
    x=np.array([[z[0]],[z[1]]])
    if modified_u:
        u=-K@x
        dx = A @ x + B * u[0]
        J=x.T@q@x+u.T*R@u
        return np.array([dx[0, 0], dx[1, 0],J[0,0]])
    else:
        u=1
        dx = A @ x + B * u
        J = x.T @ q @ x + u * R * u
        return np.array([dx[0, 0], dx[1, 0],J[0,0]])

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
        baseSystem=odeint(model,y0=[0,0,0],t=t,args=(False,))
        plt.figure('Base system')
        plt.plot(t,baseSystem[:,0],label='x1')
        plt.plot(t, baseSystem[:, 1], label='x2')
        plt.legend()
        modifiedSystem=odeint(model,y0=[1,1,0],t=t,args=(True,))
        plt.figure('Modified system')
        plt.plot(t, modifiedSystem[:, 0], label='x1')
        plt.plot(t, modifiedSystem[:, 1], label='x2')
        plt.legend()
        print('Base system J=', baseSystem[-1, 2])
        print('Modified system J=', modifiedSystem[-1, 2])
        plt.show()


def riccati(p,t):
    Pt=p.reshape((2,2))
    Pp=(Pt@A-Pt@B*R@B.T@Pt+A.T@Pt+q)
    return Pp.reshape(4)

def modelModified(z,t,k0,k1,k2,k3):
    x=np.array([[z[0]],[z[1]]])
    Pt=np.array([[k0(t),k1(t)],[k2(t),k3(t)]])
    Kt = R_inv @ B.T @ Pt
    u=-Kt@x
    dx = A @ x + B * u[0]
    J=x.T@q@x+u.T*R@u
    return np.array([dx[0, 0], dx[1, 0],J[0,0]])

def zadanie2(active):
    if active:
        t = np.linspace(0, 5, 101)
        P_t=odeint(riccati, y0=[0, 0,0,0], t=t,)
        if True:
            plt.figure('P(t)')
            plt.plot(t, P_t[:, 0], label='x[0,0]')
            plt.plot(t, P_t[:, 1], label='x[0,1]')
            plt.plot(t, P_t[:, 2], label='x[1,0]')
            plt.plot(t, P_t[:, 3], label='x[1,1]')
            plt.legend()
        Pt_inter0=scipy.interpolate.interp1d(t,P_t[:,0],fill_value='extrapolate')
        Pt_inter1 = scipy.interpolate.interp1d(t, P_t[:, 1], fill_value='extrapolate')
        Pt_inter2 = scipy.interpolate.interp1d(t, P_t[:, 2], fill_value='extrapolate')
        Pt_inter3 = scipy.interpolate.interp1d(t, P_t[:, 3], fill_value='extrapolate')
        SystemSym = odeint(modelModified, y0=[0.1, 0.1, 0], t=t, args=(Pt_inter0,Pt_inter1,Pt_inter2,Pt_inter3))
        plt.figure('Modified System with P(t)')
        plt.plot(t, SystemSym[:, 0], label='x1')
        plt.plot(t, SystemSym[:, 1], label='x2')
        plt.legend()
        print('Modified System with P(t) J=', SystemSym[-1, 2])
        plt.show()


# [x1,x2]
def modelStabilisePoint(z,t,xd1,xd2):
    x = np.array([[z[0]], [z[1]]])
    xd=np.array([[xd1],[xd2]])
    e=xd-x
    ue = -K @ e
    ud=xd1*1/c
    u=ud-ue
    dx = A @ x + B * u[0]
    #J = x.T @ q @ x + u.T * R @ u
    return np.array([dx[0, 0], dx[1, 0]])

def zadanie3(active):
    if active:
        t = np.linspace(0, 5, 101)
        stabiliseSytsem = odeint(modelStabilisePoint, y0=[0, 0], t=t, args=(1, 0))
        plt.figure('Stabilisation at point system')
        plt.plot(t, stabiliseSytsem[:, 0], label='x1')
        plt.plot(t, stabiliseSytsem[:, 1], label='x2')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    zadanie1(False)
    zadanie2(True)
    zadanie3(False)