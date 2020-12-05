import numpy as np
import matplotlib.pyplot as plt
import math
from random import random

# configure could be set here
SHAPE = 'circle'  # shape name
OFFSET = 0.02  # for preventing distance of two point is too close
FIG_SHOW = True  # visualization
OUTPUT_ANGLES = True
ANGLE_PATH = 'angles.csv'

OUTPUT_POINTS = True
POINT_PATH = 'points.csv'

OUTPUT_ERROR = True
ERROR_PATH = 'errors.csv'

OUTPUT_ITERS = True
ITER_PATH = 'iter.csv'

FIG_PATH = 'figure.png'

ARM = [1.0, 1.0, 1.0, 1.0, 1.0] # length of each arm

# optimization parameters
step_size = 0.2
iterations = 10000
minimum_distance = 0.01



# output
point_list = [] # for output
error_list = [] # for output


def custom():
    # you can put x_axis and y_axis here for drawing
    # then return [[x0, y0], [x1, y1] ... ]
    x = [[0, 0], [1, 1]]
    return x

def drawer():
    pos_list = selector_draw(SHAPE)
    joint_angles = np.zeros(5)  # init joint angles
    next_pos = np.zeros(2)  # init beginning point
    print(joint_angles)
    arm = Jointarm(ARM, joint_angles, next_pos)

    # control parameters
    isWaiting = True
    found_flag = False
    count = 0

    while True:
        prev_pos = next_pos
        next_pos = arm.next_pos
        edge_point = arm.edge_point
        errors, distance = points_distance(edge_point, next_pos)

        if isWaiting:
            if distance > minimum_distance and not found_flag:
                temp_angles, found_flag, moved_pos = inverse_kinematics(ARM, joint_angles, next_pos)
                if found_flag:
                    isWaiting = False
                else:
                    print("Not found.")
                    found_flag = False
                    arm.next_pos = [random(), random()]  # use alternative point avoiding stuck
        elif not isWaiting:
            if distance > minimum_distance and prev_pos == next_pos:
                joint_angles = joint_angles + step_size * ((temp_angles - joint_angles + np.pi) % (2 * np.pi) - np.pi)
            else:
                isWaiting = True
                found_flag = False
                arm.previous_point_list.append(moved_pos)
                arm.next_pos = pos_list[count]
                count += 1

        arm.update_joints(joint_angles)

        if count >= len(pos_list):
            print("Done, program would exit in 60 seconds\n")
            arm.fig.savefig(FIG_PATH)
            plt.pause(60)
            return




def selector_draw(name):
    # if use example
    if name == 'love':
        x = draw_love()
    elif name == 'circle':
        x = draw_circle()
    elif name == 'star':
        x = draw_star()
    else:
        x = custom()
    return x


# example shapes
def draw_love():
    x = np.linspace(-2 + OFFSET, 2 - OFFSET, 30)
    fx = np.sqrt(2 * np.sqrt(x * x) - x * x)
    gx = -1 * 2.14 * np.sqrt(math.sqrt(2) - np.sqrt(np.abs(x)))
    pos_list_up = list(zip(x, fx))
    pos_list_down = list(zip(x, gx))
    pos_list_down_reversed = pos_list_down[::-1]
    print(pos_list_up + pos_list_down_reversed)
    return pos_list_up + pos_list_down_reversed


def draw_star():
    density = 20
    x = np.linspace(-3, 3, density)
    neg_x = np.linspace(3, -3, density)
    y = np.linspace(-3, 3, density)
    neg_y = np.linspace(3, -3, density)
    pos_list_1 = list(zip(x, np.zeros(density)))
    pos_list_2 = list(zip(neg_x, np.linspace(0.02, -3, density)))
    pos_list_3 = list(zip(np.linspace(-2.98, 0, density), y))
    pos_list_4 = list(zip(np.linspace(0.02, 2.98, density), neg_y))
    pos_list_5 = list(zip(neg_x, np.linspace(-2.98, 0, density)))
    return pos_list_1 + pos_list_2 + pos_list_3 + pos_list_4 + pos_list_5


def draw_circle():
    r = 2.0
    x = np.linspace(- r + OFFSET, r - OFFSET, 40)
    fx = np.sqrt(r * r - x * x)
    pos_list_up = list(zip(x, fx))
    x_reversed = x[::-1]
    neg_fx = -1 * np.sqrt(r * r - x_reversed * x_reversed)
    pos_list_down = list(zip(x_reversed, neg_fx))
    return pos_list_up + pos_list_down


class Jointarm(object):
    def __init__(self, arm, joint_angles, next_pos):
        self.arm = np.array(arm)
        self.joint_angles = np.array(joint_angles)
        self.points = np.zeros((6, 2))

        self.lim = sum(arm)
        self.next_pos = np.array(next_pos).T

        self.previous_point_list = []

        if FIG_SHOW:  # pragma: no cover
            self.fig = plt.figure(1)
            plt.ion()
            plt.show()

        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles

        self.update_points()

    def update_points(self):
        for i in range(5):
            self.points[i + 1][0] = self.points[i][0] + self.arm[i] * np.cos(np.sum(self.joint_angles[:i + 1]))
            self.points[i + 1][1] = self.points[i][1] + self.arm[i] * np.sin(np.sum(self.joint_angles[:i + 1]))

        self.edge_point = np.array(self.points[5]).T
        if FIG_SHOW:
            self.plot()

    def plot(self):
        plt.cla()

        for i in range(6):
            if i != 5:
                plt.plot([self.points[i][0], self.points[i + 1][0]], [self.points[i][1], self.points[i + 1][1]], color = '#9A3E51', linestyle = '-')
            plt.plot(self.points[i][0], self.points[i][1], 'ko')

        plt.plot(self.next_pos[0], self.next_pos[1], 'gx')

        for i in range(1, len(self.previous_point_list)):
            plt.plot(self.previous_point_list[i][0], self.previous_point_list[i][1], color = '#9A3E51', marker = 'o')
        plt.plot(self.next_pos[0], self.next_pos[1], color = '#9A3E51', marker = 'o')
        plt.xlim([-self.lim, self.lim])
        plt.ylim([-self.lim, self.lim])
        plt.draw()
        plt.pause(0.0001)







# find joint angles that its distance less than our minimum values
# our calculation uses inverse jacobian metrix


def forward_kinematics(arm, joint_angles):
    x = y = 0
    for i in range(5):
        x += arm[i] * np.cos(np.sum(joint_angles[:i + 1]))
        y += arm[i] * np.sin(np.sum(joint_angles[:i + 1]))
    return np.array([x, y]).T

def inverse_kinematics(arm, joint_angles, next_pos):
    # Calculates the inverse kinematics using the Jacobian inverse method
    # solution given by The Pseudo Inverse Method
    for iteration in range(iterations):
        current_pos = forward_kinematics(arm, joint_angles)
        errors, distance = points_distance(current_pos, next_pos)
        if distance < minimum_distance:
            print("Solution found in %d iterations." % iteration)
            print(joint_angles)
            print(errors)
            if OUTPUT_ANGLES:
                with open(ANGLE_PATH, 'a+', encoding='utf-8') as f:
                    for i in range(len(joint_angles)):
                        f.write(str(joint_angles[i]) + ',')
                    f.write('\n')
            if OUTPUT_ERROR:
                with open(ERROR_PATH, 'a+', encoding='utf-8') as f:
                    for i in range(len(errors)):
                        f.write(str(errors[i]) + ',')
                    f.write('\n')
            if OUTPUT_POINTS:
                with open(POINT_PATH, 'a+', encoding='utf-8') as f:
                    for i in range(len(next_pos)):
                        f.write(str(next_pos[i]) + ',')
                    f.write('\n')
            if OUTPUT_ITERS:
                with open(ITER_PATH, 'a+', encoding='utf-8') as f:
                    f.write(str(iteration))
                    f.write('\n')

            return joint_angles, True, current_pos
        jacobian = jacobian_inverse(arm, joint_angles)
        joint_angles = joint_angles + np.matmul(jacobian, errors)
    return joint_angles, False, current_pos

def jacobian_inverse(arm, joint_angles):
    jacobian = np.zeros((2,5))
    for i in range(5):
        jacobian[0, i] = 0
        jacobian[1, i] = 0
        for j in range(i, 5):
            jacobian[0, i] -= arm[j] * np.sin(np.sum(joint_angles[:j]))
            jacobian[1, i] += arm[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(jacobian)


def points_distance(current_pos, next_pos):
    x_diff = next_pos[0] - current_pos[0]
    y_diff = next_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


if __name__ == '__main__':
    drawer()