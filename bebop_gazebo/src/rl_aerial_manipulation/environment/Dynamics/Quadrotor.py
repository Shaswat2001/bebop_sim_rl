import numpy as np
from math import cos,sin,tan,pi

class Quadrotor:

    def __init__(self,pos,ort,lnr_vel,ang_vel,pos_des):

        # robot parameters
        self.num_motors = 4 # number of motors on the vehicle
        self.mass = 0.506 # total mass of the vehicle, kg
        self.Ixx = 8.11858e-5  # mass-moment of inertial about x-axis, kg-m^2
        self.Iyy = 8.11858e-5  # mass-moment of inertial about y-axis, kg-m^2
        self.Izz = 6.12233e-5 # mass-moment of inertial about z-axis, kg-m^2
        self.I = np.diag([self.Ixx,self.Iyy,self.Izz])
        self.A_ref = 0.02 # reference area for drag calcs, m^2 
        self.L = 0.2 # length from body center to prop center, m
        self.k1 = 1e-7 # proportionality constant to convert motor rotational speed into thrust (T=kt*omega^2), N/(rpm)^2
        self.k2 = 1e-9 # proportionality constant to convert motor speed to torque (torque = b*omega^2), (N*m)/(rpm)^2
        self.M = np.array([[self.k1, self.k1, self.k1, self.k1],
                            [0, -self.L*self.k1, 0, self.L*self.k1],
                            [-self.L*self.k1, 0, self.L*self.k1, 0],
                            [-self.k2, self.k2, -self.k2, self.k2]])
        self.Cd = 1 # drag coefficient
        self.g = 9.8
        self.gravity = np.array([0,0,-self.g])
        self.thrust = self.mass * self.g
        self.speeds = np.ones(self.num_motors) * ((self.mass * self.g) / (self.k1 * self.num_motors)) # initial speeds of motors
        self.tau = np.zeros(3)
        self.dt = 0.01
        self.density = 1.225
        self.maxT = 16.5 #  max thrust from any single motor, N
        self.minT = 0.5 # min thrust from any single motor, N 
        self.max_angle = pi/12

        # robot state
        self.pos = pos
        self.ort = ort
        self.lnr_vel = lnr_vel
        self.ang_vel = ang_vel
        self.lnr_acc = np.zeros(self.lnr_vel.shape)
        self.ang_acc = np.zeros(self.ang_vel.shape)

        self.pos_des = pos_des
        self.ort_des = np.zeros(self.ort.shape)
        self.lnr_vel_des = np.zeros(self.lnr_vel.shape)
        self.ang_vel_des = np.zeros(self.ang_vel.shape)

        self.time = 0

    def getPoseError(self,pos):
        
        return self.pos_des - pos
    
    def getOrtError(self,ort):
        
        return self.ort_des - ort
    
    def getLnrVelError(self,lnr_vel):
        
        return self.lnr_vel_des - lnr_vel
    
    def getAngVelError(self,ang_vel):
        
        return self.ang_vel_des - ang_vel
    
    def getRotationBody2Inertia(self):
        
        phi,theta,psi = map(float,self.ort)
        RB2I = np.array([[cos(theta)*cos(psi),sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
                            [cos(theta)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)],
                            [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
        
        return RB2I

    def getRotationInertia2Body(self):

        return np.transpose(self.getRotationBody2Inertia())

    def getLnrAcc(self,force):

        RB2I = self.getRotationBody2Inertia()
        RI2B = self.getRotationInertia2Body()

        thrust_vec = RB2I @ np.array([0,0,force])

        body_vel = RI2B @ self.lnr_vel
        drag_vec_body = -self.Cd * 0.5 * self.density * self.A_ref * (body_vel)**2
        drag_vec = RB2I @ drag_vec_body

        weight = self.mass * self.gravity

        self.lnr_acc = (thrust_vec + weight + drag_vec)/self.mass

    def convertBodyRate2Omega(self):
        
        phi,theta,psi = map(float,self.ort)
        R = np.array([[1, 0, -sin(theta)],
            [0, cos(phi), cos(theta)*sin(phi)],
            [0, -sin(phi), cos(theta)*cos(phi)]])
        
        return R @ self.ang_vel

    def getOmageDot(self,tau):

        omega = self.convertBodyRate2Omega()

        omega_dot = np.linalg.inv(self.I).dot(tau - np.cross(omega,self.I @ omega))

        return omega_dot

    def convertOmageDot2AngAcc(self,omega_dot):
        
        phi,theta,psi = map(float,self.ort)
        R = np.array([[1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
                            [0, cos(phi), -sin(phi)],
                            [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])
        
        self.ang_acc = R @ omega_dot

    def convertMotorCommands(self):

        F = self.M @ self.speeds

        return F

    def getDesSpeed(self,thrust_des,tau_des):

        # Needed torque on body
        e1 = tau_des[0] * self.Ixx
        e2 = tau_des[1] * self.Iyy
        e3 = tau_des[2] * self.Izz

        #less typing
        n = self.num_motors

        # Thrust desired converted into motor speeds
        weight_speed = thrust_des / (n*self.k1)

        # Thrust differene in each motor to achieve needed torque on body
        motor_speeds = []
        motor_speeds.append(weight_speed - (e2/((n/2)*self.k1*self.L)) - (e3/(n*self.k2)))
        motor_speeds.append(weight_speed - (e1/((n/2)*self.k1*self.L)) + (e3/(n*self.k2)))
        motor_speeds.append(weight_speed + (e2/((n/2)*self.k1*self.L)) - (e3/(n*self.k2)))
        motor_speeds.append(weight_speed + (e1/((n/2)*self.k1*self.L)) + (e3/(n*self.k2)))

        # Ensure that desired thrust is within overall min and max of all motors
        thrust_all = np.array(motor_speeds) * (self.k1)
        over_max = np.argwhere(thrust_all > self.maxT)
        under_min = np.argwhere(thrust_all < self.minT)

        if over_max.size != 0:
            for i in range(over_max.size):
                motor_speeds[over_max[i][0]] = self.maxT / (self.k1)
        if under_min.size != 0:
            for i in range(under_min.size):
                motor_speeds[under_min[i][0]] = self.minT / (self.k1)
        
        self.speeds = motor_speeds

    def step(self):

        F = self.convertMotorCommands()
        self.thrust = F[0]
        self.tau = F[1:]

        self.getLnrAcc(self.thrust)

        omage_dot = self.getOmageDot(self.tau)
        self.convertOmageDot2AngAcc(omage_dot)
        
        self.lnr_vel +=  self.lnr_acc * self.dt
        self.pos += self.lnr_vel * self.dt
        self.ang_vel += self.ang_acc * self.dt
        self.ort += self.ang_vel * self.dt
        self.time += self.dt

    def stepRL(self,wheel_speed):

        self.speeds = wheel_speed
        F = self.convertMotorCommands()
        self.thrust = F[0]
        # print(F)
        self.tau = F[1:]
        
        self.getLnrAcc(self.thrust)

        omage_dot = self.getOmageDot(self.tau)
        self.convertOmageDot2AngAcc(omage_dot)
        
        self.lnr_vel +=  self.lnr_acc * self.dt
        self.pos += self.lnr_vel * self.dt
        # self.ang_vel += self.ang_acc * self.dt
        # self.ort += self.ang_vel * self.dt
        self.time += self.dt

    def set_target(self,target_pos,target_vel,target_ang_vel):

        self.pos_des = target_pos
        self.lnr_vel_des = target_vel
        self.ang_vel_des = target_ang_vel
        
    def get_states(self):

        return (self.pos,self.ort,self.lnr_vel,self.ang_vel)

    def update_states(self,action):

        self.pos += action[:3]
        self.ort += action[3:6]
        self.lnr_vel += action[6:9]
        self.ang_vel += action[9:12]

        


