import numpy as np

class Manipulator:
    def __init__(self, link1_length, link2_length):
        self.link1_length = link1_length
        self.link2_length = link2_length
        
    def dh_transform(self, alpha, a, d, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        
    def calculate_fk(self, joint_angles,UAV_position):
        alpha1, a1, d1, theta1 = 0, self.link1_length, 0, joint_angles[0]
        alpha2, a2, d2, theta2 = 0, self.link2_length, 0, joint_angles[1]
        
        transform1 = self.dh_transform(alpha1, a1, d1, theta1)
        transform2 = self.dh_transform(alpha2, a2, d2, theta2)
        
        UAV_transform = np.array([[-1,0,0,UAV_position[0]],
                                [0,0,-1,UAV_position[1]],
                                [0,-1,0,UAV_position[2]],
                                [0,0,0,1]])
        end_effector_transform = transform1 @ transform2
        world_ee_tranform = UAV_transform @ end_effector_transform
        
        position = world_ee_tranform[:3, 3]
        orientation = world_ee_tranform[:3, :3]
        
        return position, orientation
    
if __name__  == "__main__":

    # Create the manipulator object with link lengths
    link1_length = float(input("Enter length of link 1: "))
    link2_length = float(input("Enter length of link 2: "))
    manipulator = Manipulator(link1_length, link2_length)

    # Get joint angles from the user
    joint_angle1 = float(input("Enter joint angle 1 (radians): "))
    joint_angle2 = float(input("Enter joint angle 2 (radians): "))
    joint_angles = [joint_angle1, joint_angle2]

    world = np.array([[-1,0,0,0.1],
                      [0,0,-1,0.1],
                      [0,-1,0,0.1],
                      [0,0,0,1]])
    # Calculate forward kinematics
    end_effector_position, end_effector_orientation = manipulator.calculate_fk(joint_angles,world)

    print("\nEnd Effector Position:")
    print(end_effector_position)
    print("\nEnd Effector Orientation (Rotation Matrix):")
    print(end_effector_orientation)