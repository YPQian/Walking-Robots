import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class Biped3D(object):
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=60,
                                     cameraPitch=-30, cameraTargetPosition=[0, 0, 0.2])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        self.ground = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("/yanshee/robots/yanshee.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=False)
        self.joints = self.get_joints()
        self.n_j = len(self.joints)
        self.simu_f = 500 # Simulation frequency, Hz
        p.setTimeStep(1.0/self.simu_f)
        self.motion_f = 2  # Controlled motion frequency, Hz
        self.q_vec = np.zeros(self.n_j)
        self.dq_vec = np.zeros(self.n_j)
        self.q_mat = np.zeros((self.simu_f * 3, self.n_j))
        self.q_d_mat = np.zeros((self.simu_f * 3, self.n_j))
        self.init_plot()


    def run(self):
        for i in range(int(5e3)):
            t = i / self.simu_f
            torque_array = self.controller(t)
            self.q_vec, self.dq_vec = self.step(torque_array)
            if 0 == i % 20:
                self.update_plot()
            time.sleep(1/self.simu_f)
        p.disconnect()

    def step(self, torque_array):
        self.set_motor_torque_array(torque_array)
        p.stepSimulation()
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec
        return self.get_joint_states()

    def get_joints(self):
        all_joints = []
        for j in range(p.getNumJoints(self.robot)):
            # Disable motor in order to use direct torque control.
            info = p.getJointInfo(self.robot, j)
            joint_type = info[2]
            if (joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE):
                all_joints.append(j)
                p.setJointMotorControl2(self.robot, j,
                                        controlMode=p.POSITION_CONTROL, force=0)
        joints = all_joints[0:]
        # print("Number of All Joints:", p.getNumJoints(self.robot))
        # print("Number of All Revolute Joints:", joints)
        return joints

    def get_joint_states(self):
        '''
        :return: q_vec: joint angle, dq_vec: joint angular velocity
        '''
        q_vec = np.zeros(self.n_j)
        dq_vec = np.zeros(self.n_j)
        for j in range(self.n_j):
            q_vec[j], dq_vec[j], _, _  = p.getJointState(self.robot, self.joints[j])
        return q_vec, dq_vec

    def set_motor_torque_array(self, torque_array = None):
        '''
        :param torque_array: the torque of [rHAA, rHFE, rKFE]
        :Specifically, H: Hip, K: Knee, AA:abduction/adduction, FE:flexion/extension
        '''
        if torque_array is None:
            torque_array = np.zeros(self.n_j)
        for j in range(len(self.joints)):
            p.setJointMotorControl2(self.robot, self.joints[j], p.TORQUE_CONTROL, force=torque_array[j])

    def controller(self, t, type='joint'):
        if 'joint' == type:
            return self.joint_controller(t)

    def joint_controller(self, t):
        a = np.deg2rad(30)
        q_d_haa = 0
        q_d_hfe = a*np.sin(2*np.pi*self.motion_f*t)
        q_d_kfe = a/2.0*(np.cos(2*np.pi*self.motion_f*t)+1)
        q_d_vec = np.array([q_d_haa, q_d_hfe, q_d_kfe])
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = q_d_vec
        dq_d_haa = 0
        dq_d_hfe = 2*np.pi*self.motion_f*a*np.cos(2*np.pi*self.motion_f*t)
        dq_d_kfe = -np.pi*self.motion_f*a*np.sin(2*np.pi*self.motion_f*t)
        dq_d_vec = np.array([dq_d_haa, dq_d_hfe, dq_d_kfe])

        # PD controller gains
        k_vec = np.array([100, 50, 50])
        b_vec = np.array([0.05, 0.05, 0.05])
        return self.joint_PD_controller(k_vec, b_vec, self.q_vec, self.dq_vec, q_d_vec, dq_d_vec)

    def joint_PD_controller(self, k_vec, b_vec, q_vec, dq_vec, q_d_vec, dq_d_vec):
        return k_vec*(q_d_vec-q_vec) + b_vec*(dq_d_vec-dq_vec)

    def init_plot(self):
        self.fig = plt.figure(figsize=(5, 9))
        joint_names = ['rHAA', 'rHFE', 'rKFE', ]
        self.q_d_lines = []
        self.q_lines = []
        for i in range(3):
            plt.subplot(3, 1, i+1)
            q_d_line, = plt.plot(self.q_d_mat[:, i], '-')
            q_line, = plt.plot(self.q_mat[:, i], '--')
            self.q_d_lines.append(q_d_line)
            self.q_lines.append(q_line)
            plt.ylabel('q_{} (rad)'.format(joint_names[i]))
            plt.ylim([-1.5, 1.5])
        plt.xlabel('Simulation steps')
        self.fig.legend(['q_d', 'q'], loc='lower center', ncol=2, bbox_to_anchor=(0.49, 0.97), frameon=False)
        self.fig.tight_layout()
        plt.draw()

    def update_plot(self):
        for i in range(3):
            self.q_d_lines[i].set_ydata(self.q_d_mat[:, i])
            self.q_lines[i].set_ydata(self.q_mat[:, i])
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    robot = Biped3D()
    robot.run()