import numpy as np
import copy
from math import cos,sin,tan,pi
from scipy.linalg import expm
from scipy.integrate import odeint,ode
from .Cmatrix import C_Matrix_vars
import threading
from scipy.integrate import solve_ivp

class UAM:

    def __init__(self,publish_func,min_bound,max_bound):

        # robot parameters
        self.num_motors = 6 # number of motors on the vehicle
        self.m_b = 4.2 # total mass of the vehicle, kg
        self.Ixx = 0.430475  # mass-moment of inertial about x-axis, kg-m^2
        self.Iyy = 0.430587  # mass-moment of inertial about y-axis, kg-m^2
        self.Izz = 0.592651 # mass-moment of inertial about z-axis, kg-m^2
        self.I = np.diag([self.Ixx,self.Iyy,self.Izz])

        self.lmd = 2*np.identity(10)
        self.M_hat = np.identity(10)
        self.C_hat = np.identity(10)
        self.G_hat = np.array([[0,0,-10,0,0,0,0,0,0,0]]).T

        self.K = 8*np.identity(10)
        self.A = 10*np.identity(10)
        self.K_A = 10*np.identity(10)

        self.publish_func = publish_func

        self.dt = 0.000001
        self.m_l = [0.10483260,0.14234630,0.13467049,0.23550927] # total mass of the vehicle, kg
        self.length = [0.077,0.13,0.124,0.126]
        self.I_l = []
        for i in range(len(self.m_l)):

            mass = self.m_l[i]
            length = self.length[i]
            Ixx = (1/12)*mass*length**2

            if i == 2:
                Ixx = (1/13)*mass*(length**2)
            elif i == 3:
                Ixx = (1/14)*mass*(length**2)
            Iyy = Ixx
            Izz = 0
            self.I_l.append(np.diag([Ixx,Iyy,Izz]))

        # robot state
        self.q = None
        self.qdot = None
        self.qddot = None

        self.q_des = None
        self.qdot_des = None
        self.qddot_des = None

        self.time = 0

        self.min_bound = min_bound
        self.max_bound = max_bound

    def getPoseError(self,position):
        
        return position - self.q_des
    
    def getVelError(self,velocity):
        
        return velocity - self.qdot_des

    def hat(self,vector):

        return np.array([[0,-vector[2][0],vector[1][0]],
                            [vector[2][0],0,-vector[0][0]],
                            [-vector[1][0],vector[0][0],0]])

    def inverse_hat(self,matrix):

        return np.array([-matrix[1,2],matrix[0,2],-matrix[0,1]]).reshape(-1,1)
        
    def odefunc(self,t, y):
        
        delta_hat = np.array(y[0:10]).reshape(-1,1)
        vel = np.array(y[10:20]).reshape(-1,1)
        pos = np.array(y[20:30]).reshape(-1,1)

        self.publish_func(np.clip(pos[:,0],self.min_bound,self.max_bound))

        pos_error = self.getPoseError(pos)
        vel_error = self.getVelError(vel)
        s = vel_error + self.lmd @ pos_error
        qrDot = self.qdot_des - self.lmd @ pos_error
        qrDotDot = self.qddot_des - self.lmd @ vel_error
        tau = self.M_hat @ qrDotDot + self.C_hat @ qrDot + self.G_hat + delta_hat - self.A @ s  -self.K @ np.tanh(s)
        delta_hatDot = - self.K_A @ s
        acceleration = self.step(t, tau, pos,vel)
        dt = 0.01001001001001001
        dydt = list(np.concatenate((delta_hatDot,acceleration, vel)).T[0])
        
        return dydt
    
    def step(self,time,tau,q,qdot):
        
        x,y,z = q[0:3,0]
        phi,theta,psi = q[3:6,0]
        n1,n2,n3,n4 = q[6:10,0]

        x_dot,y_dot,z_dot = qdot[0:3,0]
        phi_dot,theta_dot,psi_dot = qdot[3:6,0]
        n1_dot,n2_dot,n3_dot,n4_dot = qdot[6:10,0]

        p_b = np.array([x,y,z]).reshape(-1,1)
        Tb = np.array([[0,-sin(psi),cos(psi)*cos(theta)],
                       [0,cos(psi), cos(theta)*sin(psi)],
                       [1, 0, -sin(theta)]])
        
        Rb = np.array([[cos(psi)*cos(theta), cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi), sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta)],
                       [sin(psi)*cos(theta), sin(psi)*sin(phi)*sin(theta) + cos(phi)*cos(psi), -sin(phi)*cos(psi)+cos(phi)*sin(psi)*sin(theta)],
                       [-sin(theta), cos(theta)*sin(phi), cos(phi)*cos(theta)]])
        
        Q = Rb.T @ Tb

        h1 = np.array([0,0,1]).reshape(-1,1)
        h1_b = np.array([0,0,1]).reshape(-1,1)
        P10 = np.array([0,0,0]).reshape(-1,1)
        p1c = np.array([0,0,self.length[0]/2]).reshape(-1,1)

        R01 = expm(self.hat(h1) * n1)
        p1c_b = P10 +  R01 @ p1c
        h2 = np.array([0,1,0]).reshape(-1,1)
        h2_b = R01 @ h2
        p12 = np.array([0,0,self.length[0]]).reshape(-1,1)
        p2c = np.array([0,0,self.length[1]/2]).reshape(-1,1)
        R12 = expm(self.hat(h2) * n2)
        R02 = R01 @ R12
        p2c_b = P10 +  R01 @ p12 + R02 @ p2c

        h3 = np.array([0,1,0]).reshape(-1,1)
        h3_b = R02 @ h3
        p23 = np.array([0,0,self.length[1]]).reshape(-1,1)
        p3c = np.array([0,0,self.length[2]/2]).reshape(-1,1)

        R23 = expm(self.hat(h3) * n3)
        R03 = R02 @ R23
        p3c_b = P10 +  R01 @ p12 + R02@p23 + R03@p3c

        h4 = np.array([0,1,0]).reshape(-1,1)
        h4_b = R03 @ h4
        p34 = np.array([0,0,self.length[2]]).reshape(-1,1)
        p4T = np.array([0,0,self.length[3]]).reshape(-1,1)
        p4c = np.array([0,0,self.length[3]/2]).reshape(-1,1)

        R34 = expm(self.hat(h4) * n4)
        R04 = R03 @ R34
        p4c_b = P10 +  R01 @ p12 + R02 @ p23 + R03 @ p34 + R04 @ p4c

        p1_b = P10 + R01 @ p12
        p2_b = P10 + R01 @ p12 + R02 @ p23
        p3_b = P10 + R01 @ p12 + R02 @ p23 + R03 @ p34
        p4_b = P10 + R01 @ p12 + R02 @ p23 + R03 @ p34 + R04 @ p4T
        Pee = p_b  + Rb @ p4_b

        Jw1c = np.concatenate([h1_b,np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))],axis=-1)
        Jv1c = np.concatenate([self.hat(h1_b) @ (p1c_b - P10),np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))],axis=-1)

        Jw2c = np.concatenate([h1_b,h2_b,np.zeros((3,1)),np.zeros((3,1))],axis=-1)
        Jv2c = np.concatenate([self.hat(h1_b) @ (p2c_b - P10),self.hat(h2_b) @ (p2c_b - p1_b),np.zeros((3,1)),np.zeros((3,1))],axis=-1)

        Jw3c = np.concatenate([h1_b,h2_b,h3_b,np.zeros((3,1))],axis=-1)
        Jv3c = np.concatenate([self.hat(h1_b) @ (p3c_b - P10),self.hat(h2_b) @ (p3c_b - p1_b),self.hat(h3_b) @ (p3c_b - p2_b),np.zeros((3,1))],axis=-1)

        Jw4c = np.concatenate([h1_b,h2_b,h3_b,h4_b],axis=-1)
        Jv4c = np.concatenate([self.hat(h1_b) @ (p4c_b - P10),self.hat(h2_b) @ (p4c_b - p1_b),self.hat(h3_b) @ (p4c_b - p2_b),self.hat(h4_b) @ (p4c_b - p3_b)],axis=-1)

        Jt_h = np.concatenate([h1_b,h2_b,h3_b,h4_b],axis=-1)
        Jt_p = np.concatenate([self.hat(h1_b) @ (p4_b- P10),self.hat(h2_b) @ (p4_b- p1_b),self.hat(h3_b) @ (p4_b- p2_b),self.hat(h4_b) @ (p4_b- p3_b)],axis=-1)

        Jt = np.concatenate([Jt_h,Jt_p],axis=0)

        JwT = Jt[0:3,0:4]
        JvT = Jt[3:6,0:4]

        M11 = (np.sum(self.m_l) +  self.m_b)*np.identity(3)
        M22 = Q.T @ self.I @ Q + \
                self.m_l[0]*Tb.T @ self.hat((Rb @ p1c_b)) @ self.hat(Rb @ p1c_b) @ Tb + Q.T @ R01 @ self.I_l[0] @ R01.T @ Q + \
                self.m_l[1]*Tb.T @ self.hat((Rb @ p2c_b)) @ self.hat(Rb @ p2c_b) @ Tb + Q.T @ R02 @ self.I_l[1] @ R02.T @ Q + \
                self.m_l[2]*Tb.T @ self.hat((Rb @ p3c_b)) @ self.hat(Rb @ p3c_b) @ Tb + Q.T @ R03 @ self.I_l[2] @ R03.T @ Q + \
                self.m_l[3]*Tb.T @ self.hat((Rb @ p4c_b)) @ self.hat(Rb @ p4c_b) @ Tb + Q.T @ R04 @ self.I_l[3] @ R04.T @ Q

        M33 = self.m_l[0]*Jv1c.T @ Jv1c + Jw1c.T @ R01 @ self.I_l[0] @ R01.T @ Jw1c + \
                self.m_l[1]*Jv2c.T @ Jv2c + Jw2c.T @ R02 @ self.I_l[1] @ R02.T @ Jw2c + \
                self.m_l[2]*Jv3c.T @ Jv3c + Jw3c.T @ R03 @ self.I_l[2] @ R03.T @ Jw3c + \
                self.m_l[3]*Jv4c.T @ Jv4c + Jw4c.T @ R04 @ self.I_l[3] @ R04.T @ Jw4c
        
        M12 = -self.m_l[0]*self.hat((Rb @ p1c_b)) - \
                self.m_l[1]*self.hat((Rb @ p2c_b)) - \
                self.m_l[2]*self.hat((Rb @ p3c_b)) - \
                self.m_l[3]*self.hat((Rb @ p4c_b))
        
        M21 = M12.T
        M13 = self.m_l[0]*Rb @ Jv1c + self.m_l[1]*Rb @ Jv2c + self.m_l[2]*Rb @ Jv3c + self.m_l[3]*Rb @ Jv4c
        M31 = M13.T

        M23 = Q.T @ R01 @ self.I_l[0] @ R01.T @ Jw1c - self.m_l[0]*Tb.T @ self.hat((Rb @ p1c_b)) @ Rb @ Jv1c + \
                Q.T @ R02 @ self.I_l[1] @ R02.T @ Jw2c - self.m_l[1]*Tb.T @ self.hat((Rb @ p2c_b)) @ Rb @ Jv2c + \
                Q.T @ R03 @ self.I_l[2] @ R03.T @ Jw3c - self.m_l[2]*Tb.T @ self.hat((Rb @ p3c_b)) @ Rb @ Jv3c + \
                Q.T @ R04 @ self.I_l[3] @ R04.T @ Jw4c - self.m_l[3]*Tb.T @ self.hat((Rb @ p4c_b)) @ Rb @ Jv4c
        
        M32 = M23.T

        M1 = np.concatenate([M11,M12,M13],axis=1)
        M2 = np.concatenate([M21,M22,M23],axis=1)
        M3 = np.concatenate([M31,M32,M33],axis=1)

        M = np.concatenate([M1,M2,M3],axis=0)
        C = C_Matrix_vars(self.m_b,self.m_l[0],self.m_l[1],self.m_l[2],self.m_l[3],self.Ixx,self.Iyy,self.Izz,self.I_l[0][0][0],self.I_l[0][1][1],self.I_l[0][2][2],self.I_l[1][0][0],self.I_l[1][1][1],self.I_l[1][2][2],self.I_l[2][0][0],self.I_l[2][1][1],self.I_l[2][2][2],self.I_l[3][0][0],self.I_l[3][1][1],self.I_l[3][2][2],self.length[0],self.length[1],self.length[2],self.length[3],x,y,z,phi,theta,psi,n1,n2,n3,n4,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot,n1_dot,n2_dot,n3_dot,n4_dot)

        G1 = 0
        G2 = 0
        G3 = (49*self.m_l[0])/5 + (49*self.m_l[1])/5 + (49*self.m_l[2])/5 + (49*self.m_l[3])/5 + (49*self.m_b)/5
        G4 = (49*self.m_l[2]*(cos(phi)*cos(theta)*((self.length[2]*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)))/2 + self.length[1]*sin(n1)*sin(n2)) - cos(theta)*sin(phi)*(self.length[0] + (self.length[2]*(cos(n2)*cos(n3) - sin(n2)*sin(n3)))/2 + self.length[1]*cos(n2))))/5 + (49*self.m_l[3]*(cos(phi)*cos(theta)*(self.length[2]*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)) + (self.length[3]*(cos(n4)*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)) - sin(n4)*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1))))/2 + self.length[1]*sin(n1)*sin(n2)) - cos(theta)*sin(phi)*(self.length[0] + self.length[2]*(cos(n2)*cos(n3) - sin(n2)*sin(n3)) + self.length[1]*cos(n2) + (self.length[3]*(cos(n4)*(cos(n2)*cos(n3) - sin(n2)*sin(n3)) - sin(n4)*(cos(n2)*sin(n3) + cos(n3)*sin(n2))))/2)))/5 - (49*self.m_l[1]*(cos(theta)*sin(phi)*(self.length[0] + (self.length[1]*cos(n2))/2) - (self.length[1]*cos(phi)*cos(theta)*sin(n1)*sin(n2))/2))/5 - (49*self.length[0]*self.m_l[0]*cos(theta)*sin(phi))/10
        G5 = -(49*self.m_l[2]*(cos(theta)*((self.length[2]*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2)))/2 + self.length[1]*cos(n1)*sin(n2)) + sin(phi)*sin(theta)*((self.length[2]*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)))/2 + self.length[1]*sin(n1)*sin(n2)) + cos(phi)*sin(theta)*(self.length[0] + (self.length[2]*(cos(n2)*cos(n3) - sin(n2)*sin(n3)))/2 + self.length[1]*cos(n2))))/5 - (49*self.m_l[1]*(cos(phi)*sin(theta)*(self.length[0] + (self.length[1]*cos(n2))/2) + (self.length[1]*cos(n1)*cos(theta)*sin(n2))/2 + (self.length[1]*sin(n1)*sin(n2)*sin(phi)*sin(theta))/2))/5 - (49*self.m_l[3]*(cos(theta)*((self.length[3]*(cos(n4)*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2)) + sin(n4)*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3))))/2 + self.length[2]*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2)) + self.length[1]*cos(n1)*sin(n2)) + sin(phi)*sin(theta)*(self.length[2]*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)) + (self.length[3]*(cos(n4)*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)) - sin(n4)*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1))))/2 + self.length[1]*sin(n1)*sin(n2)) + cos(phi)*sin(theta)*(self.length[0] + self.length[2]*(cos(n2)*cos(n3) - sin(n2)*sin(n3)) + self.length[1]*cos(n2) + (self.length[3]*(cos(n4)*(cos(n2)*cos(n3) - sin(n2)*sin(n3)) - sin(n4)*(cos(n2)*sin(n3) + cos(n3)*sin(n2))))/2)))/5 - (49*self.length[0]*self.m_l[0]*cos(phi)*sin(theta))/10
        G6 = 0
        G7 = (49*self.m_l[2]*(sin(theta)*((self.length[2]*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)))/2 + self.length[1]*sin(n1)*sin(n2)) + cos(theta)*sin(phi)*((self.length[2]*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2)))/2 + self.length[1]*cos(n1)*sin(n2))))/5 + (49*self.m_l[3]*(sin(theta)*(self.length[2]*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)) + (self.length[3]*(cos(n4)*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2)) - sin(n4)*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1))))/2 + self.length[1]*sin(n1)*sin(n2)) + cos(theta)*sin(phi)*((self.length[3]*(cos(n4)*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2)) + sin(n4)*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3))))/2 + self.length[2]*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2)) + self.length[1]*cos(n1)*sin(n2))))/5 + (49*self.m_l[1]*((self.length[1]*sin(n1)*sin(n2)*sin(theta))/2 + (self.length[1]*cos(n1)*cos(theta)*sin(n2)*sin(phi))/2))/5
        G8 = - (49*self.m_l[1]*((self.length[1]*cos(n1)*cos(n2)*sin(theta))/2 + (self.length[1]*cos(phi)*cos(theta)*sin(n2))/2 - (self.length[1]*cos(n2)*cos(theta)*sin(n1)*sin(phi))/2))/5 - (49*self.m_l[3]*(sin(theta)*((self.length[3]*(cos(n4)*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3)) - sin(n4)*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2))))/2 + self.length[2]*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3)) + self.length[1]*cos(n1)*cos(n2)) + cos(phi)*cos(theta)*(self.length[2]*(cos(n2)*sin(n3) + cos(n3)*sin(n2)) + (self.length[3]*(cos(n4)*(cos(n2)*sin(n3) + cos(n3)*sin(n2)) + sin(n4)*(cos(n2)*cos(n3) - sin(n2)*sin(n3))))/2 + self.length[1]*sin(n2)) + cos(theta)*sin(phi)*(self.length[2]*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1)) + (self.length[3]*(cos(n4)*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1)) + sin(n4)*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2))))/2 - self.length[1]*cos(n2)*sin(n1))))/5 - (49*self.m_l[2]*(sin(theta)*((self.length[2]*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3)))/2 + self.length[1]*cos(n1)*cos(n2)) + cos(theta)*sin(phi)*((self.length[2]*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1)))/2 - self.length[1]*cos(n2)*sin(n1)) + cos(phi)*cos(theta)*((self.length[2]*(cos(n2)*sin(n3) + cos(n3)*sin(n2)))/2 + self.length[1]*sin(n2))))/5
        G9 = - (49*self.m_l[3]*(sin(theta)*((self.length[3]*(cos(n4)*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3)) - sin(n4)*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2))))/2 + self.length[2]*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3))) + cos(phi)*cos(theta)*(self.length[2]*(cos(n2)*sin(n3) + cos(n3)*sin(n2)) + (self.length[3]*(cos(n4)*(cos(n2)*sin(n3) + cos(n3)*sin(n2)) + sin(n4)*(cos(n2)*cos(n3) - sin(n2)*sin(n3))))/2) + cos(theta)*sin(phi)*(self.length[2]*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1)) + (self.length[3]*(cos(n4)*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1)) + sin(n4)*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2))))/2)))/5 - (49*self.m_l[2]*((self.length[2]*sin(theta)*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3)))/2 + (self.length[2]*cos(theta)*sin(phi)*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1)))/2 + (self.length[2]*cos(phi)*cos(theta)*(cos(n2)*sin(n3) + cos(n3)*sin(n2)))/2))/5
        G10 = -(49*self.m_l[3]*((self.length[3]*sin(theta)*(cos(n4)*(cos(n1)*cos(n2)*cos(n3) - cos(n1)*sin(n2)*sin(n3)) - sin(n4)*(cos(n1)*cos(n2)*sin(n3) + cos(n1)*cos(n3)*sin(n2))))/2 + (self.length[3]*cos(phi)*cos(theta)*(cos(n4)*(cos(n2)*sin(n3) + cos(n3)*sin(n2)) + sin(n4)*(cos(n2)*cos(n3) - sin(n2)*sin(n3))))/2 + (self.length[3]*cos(theta)*sin(phi)*(cos(n4)*(sin(n1)*sin(n2)*sin(n3) - cos(n2)*cos(n3)*sin(n1)) + sin(n4)*(cos(n2)*sin(n1)*sin(n3) + cos(n3)*sin(n1)*sin(n2))))/2))/5

        G = np.array([[G1,G2,G3,G4,G5,G6,G7,G8,G9,G10]]).T

        term1 = -self.hat(Rb @ p4_b) @ Tb
        term2 = Rb @ JvT
        term3 = Rb @ JwT

        J_a1 = np.concatenate([np.identity(3),np.zeros((3,3)),np.zeros((3,4))],axis=1)
        J_a2 = np.concatenate([np.zeros((3,3)),Tb,np.zeros((3,4))],axis=1)
        J_a3 = np.concatenate([np.identity(3),term1,term2],axis=1)
        J_a4 = np.concatenate([np.zeros((3,3)),Tb,term3],axis=1)

        J_a = np.concatenate([J_a1,J_a2,J_a3,J_a4],axis=0)

        z_ext = -10
        f_ext = np.array([0,0,sin(5*time),0,0,0,0,0,z_ext,0,0,0]).reshape(-1,1)
        tau_ext = J_a.T @ f_ext

        qdotdot = np.linalg.inv(M) @ (tau + tau_ext - C @ qdot - G)
        
        return qdotdot
        
    def solve(self,q_des,qdot_des):

        self.q_des = q_des.reshape(-1,1)
        self.qdot_des = qdot_des.reshape(-1,1)

        self.q = self.q.reshape(-1,1)
        self.qdot = self.qdot.reshape(-1,1)
        self.qddot = self.qddot.reshape(-1,1)
        
        initial_time = 0
        final_time = 10  # Adjust as needed
        tspan = np.linspace(initial_time, final_time, num=1000)  
        initial_velocity = self.qdot
        initial_position = self.q
        delta_hat = np.array([[0.01]*10]).T
        
        y0 = list(np.concatenate((delta_hat,initial_velocity, initial_position)).T[0])

        # solution = odeint(self.odefunc, y0, tspan)
        solution = solve_ivp(self.odefunc, (initial_time, final_time), y0, method='BDF')

        self.qdot = solution.y[10:20,-1]
        self.q = np.clip(solution.y[20:30,-1],self.min_bound,self.max_bound)
        
        return self.q,self.qdot

    def reset(self,q,qdot,qddot):
        self.q = q
        self.qdot = qdot
        self.qddot = qddot

        self.q_des = None
        self.qdot_des = None
        self.qddot_des = np.zeros((self.qddot.shape[0],1))

if __name__ == "__main__":

    pos_desired = np.array([[2,3,10,0,0,0,0,0,0.5,0.5]]).T
    # pos_desired = np.array([[0.0]*10]).T
    vel_desired = np.array([[0.0]*10]).T
    accl_desired = np.array([[0.0]*10]).T
    current_pose = np.array([[0.01]*10]).T
    # current_pose = np.array([[1,2,3,4,5,6,7,8,9,10]],dtype=np.float64).T
    # current_vel = np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]).T
    current_vel = np.array([[0.01]*10]).T
    Delta_hat = np.array([[0.01]*10]).T
    uam = UAM()
    uam.reset(current_pose,current_vel,np.array([[0.01]*10]).T)
    uam.solve(pos_desired,vel_desired)

    # thread = threading.Thread(target=uam.ode_method,args=())
    # thread.start()
