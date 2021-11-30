import distutils.util
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import time

# assuming running from raycasting-simulation/Automator
sys.path.append("../PycastWorld")

from math import acos, asin, atan, cos, sin, tan, pi
from math import floor
from math import radians
from pycaster import PycastWorld, Turn, Walk
from numpy.random import default_rng
sys.path.append("/home/eoca2018/unrealcv/client/python")
from unrealcv import client as ue4
from unrealcv.util import read_png
from UnrealUtils import UE4EnvWrapper

rng = default_rng()

# NOISE CONTROL
# the standard deviation of the Gaussian that random angles are drawn from
rand_angle_scale = pi / 36  # 5 degree s.d.

# the minimum of the uniform distribution that random distances (to move) are drawn from
rand_step_scale = 0.4

enws = {"Dir.EAST": 180, "Dir.NORTH": 90, "Dir.WEST": 0, "Dir.SOUTH": 270}


def in_targ_cell(base_dir, c_targ_x, c_targ_y, x, y):
    if base_dir == 0 or base_dir == 180:
        if abs(c_targ_x - x) < 0.4:
            return True
    else:
        if abs(c_targ_y - y) < 0.4:
            return True
    return False


class Driver:
    def __init__(
        self, c_targ_x, c_targ_y, base_dir, targ_dir, world, img_dir=None, show_freq=0,
    ):
        self.c_targ_x = c_targ_x
        self.c_targ_y = c_targ_y
        self.base_dir = base_dir
        self.targ_dir = targ_dir

        self.world = world
        self.curr_x = self.world.get_x()
        self.curr_y = self.world.get_y()

        self.direction = 0
        self.update_direction()

        self.dist = math.inf
        self.update_dist()

        self.angle = 0
        self.step = math.inf

        self.img_dir = img_dir
        if self.img_dir != None:
            stack_conds = []
            stack_conds.append(os.path.isdir(os.path.join(img_dir, "left")))
            stack_conds.append(os.path.isdir(os.path.join(img_dir, "right")))
            stack_conds.append(os.path.isdir(os.path.join(img_dir, "straight")))

            # if subdirectories exist, then stacking method not used
            if all(stack_conds):
                self.img_num_l = len(os.listdir(os.path.join(img_dir, "left")))
                self.img_num_r = len(os.listdir(os.path.join(img_dir, "right")))
                self.img_num_s = len(os.listdir(os.path.join(img_dir, "straight")))
                self.stack_dir = False
            else:
                self.img_num = len(os.listdir(img_dir))
                self.stack_dir = True

        self.show_freq = show_freq

    def update_dist(self):
        self.dist = math.sqrt(
            (self.c_targ_x - self.world.get_x()) ** 2
            + (self.c_targ_y - self.world.get_y()) ** 2
        )
#         print(f"dist: {self.dist}")

    def update_direction(self):
        if not -1 <= self.world.get_dir_x() <= 1:
            dir_x = round(self.world.get_dir_x())
        else:
            dir_x = self.world.get_dir_x()

        if not -1 <= self.world.get_dir_y() <= 1:
            dir_y = round(self.world.get_dir_y())
        else:
            dir_y = self.world.get_dir_y()

        if dir_x > 0 and dir_y >= 0:
            dir = acos(dir_x)
        elif dir_x <= 0 and dir_y >= 0:
            dir = acos(dir_x)
        elif dir_x < 0 and dir_y < 0:
            dir = pi - asin(dir_y)
        elif dir_x >= 0 and dir_y < 0:
            dir = asin(dir_y)

        self.direction = dir % (2 * pi)

    # adjust for smoother path
    def modified_targ(self, delta):
        if self.base_dir == 0 or self.base_dir == 180:
            if self.targ_dir == 90:
                return self.c_targ_x, self.c_targ_y + delta
            elif self.targ_dir == 270:
                return self.c_targ_x, self.c_targ_y - delta
        elif self.base_dir == 90 or self.base_dir == 270:
            if self.targ_dir == 0:
                return self.c_targ_x + delta, self.c_targ_y
#                 return self.c_targ_x - delta, self.c_targ_y
            elif self.targ_dir == 180:
                return self.c_targ_x - delta, self.c_targ_y
#                 return self.c_targ_x + delta, self.c_targ_y
        return self.c_targ_x, self.c_targ_y

    def get_angle(self):
        print("GETTING ANGLE")
        mod_x, mod_y = self.modified_targ(0.15)
        print(f"MOD_X: {mod_x}, MOD_Y: {mod_y}")
        print(f"CURR_X: {self.curr_x}, CURR_Y: {self.curr_y}")
        # top right? 
        # top right -> top left
        if self.curr_x <= mod_x and self.curr_y <= mod_y:
            print("in top right")
            if mod_x == self.curr_x:
                theta = pi / 2
            else:
                theta = (atan((mod_y - self.curr_y) / (mod_x - self.curr_x))) % (2 * pi) 

        # case where target pos is up and to the left
        elif self.curr_x > mod_x and self.curr_y <= mod_y:
            print("in top left")
            if mod_y == self.curr_y:
                theta = pi
            else:
                theta = (atan((self.curr_x - mod_x) / (mod_y - self.curr_y))) % (
                    2 * pi
                ) + pi / 2

        # case where target pos is down and to the left
        elif self.curr_x > mod_x and self.curr_y > mod_y:
            print("in bottom left")
            if mod_x == self.curr_x:
                theta = 0#3 * pi / 2
            else:
                theta = (atan((self.curr_y - mod_y) / (self.curr_x - mod_x))) % (
                    2 * pi
                ) + pi

        # case where target pos is down and to the right
        else:
            print("in bottom right")
            if self.curr_y == mod_y:
                theta = 0
            else:
                theta = (atan((mod_x - self.curr_x) / (self.curr_y - mod_y))) % (
                    2 * pi
                ) + 3 * pi / 2
        print(f"THETA {theta*(180/pi)}")
        return theta

    def set_rand_angle(self):
        theta = self.get_angle()
        self.angle = rng.normal(loc=theta, scale=rand_angle_scale) % (2 * pi)
        print(f"self.angle: {self.angle}")

    def set_rand_step(self):
        self.step = rng.uniform(rand_step_scale, self.dist_to_wall())

    def abs_angle_diff(self, angle):
        print(f"ABS_ANGLE_DIFF: dir: {self.direction}, angle: {angle}")
        abs_diff = abs(self.direction - angle)
        return abs_diff % (2 * pi)

    def turn_right(self, angle):
#         print(f"right condition: dir {self.direction*(180/math.pi)} angle: {angle*(180/math.pi)}")
#         print(f"adje right condition: dir {self.direction*(180/math.pi)} angle: {angle*(180/math.pi)}")
        adj_dir = self.direction#(pi - self.direction)
        if adj_dir > angle:
            if adj_dir - angle > pi:
                return False
            else:
                return True
        else:
            if angle - adj_dir > pi:
                return True
            else:
                return False
            
    @staticmethod
    def filename_from_angle_deg(angle: float, i: int) -> str:
        return f"{i:>06}_{angle:.3f}".replace(".", "p") + "_" + str(round(time.time() * 1000)) + "_" + ".png"

    def turn_to_angle(self):
#         self.world.forward()
        i = 0
        prev_turn = None
        while self.abs_angle_diff(self.angle) > 0.1:
            if self.turn_right(self.angle):

                if prev_turn == "left":
                    print("no left to right allowed")
                    break

                # save image right
                agent_dir = -abs(90 + self.world.get_angle())
                angle_label = self.filename_from_angle_deg(agent_dir, self.img_num)
                if self.img_dir != None:
                    if self.stack_dir:
                        self.world.save_png(
                            os.path.join(self.img_dir, f"{angle_label}")
                        )
                        self.img_num += 1
                    else:
                        self.world.save_png(
                            os.path.join(
                                self.img_dir, "right", f"{self.img_num_r:05}.png"
                            )
                        )
                        self.img_num_r += 1
                print("turning right")
                self.world.right()             
                prev_turn = "right"

            else:
                if prev_turn == "right":
                    print("no right to left allowed")
                    break

                # save image left
                agent_dir = abs(90 + self.world.get_angle())
                angle_label = self.filename_from_angle_deg(agent_dir, self.img_num)
                if self.img_dir != None:
                    if self.stack_dir:
                        self.world.save_png(
                            os.path.join(self.img_dir, f"{angle_label}")
                        )
                        self.img_num += 1
                    else:
                        self.world.save_png(
                            os.path.join(
                                self.img_dir, "left", f"{self.img_num_l:05}.png"
                            )
                        )
                        self.img_num_l += 1
                print("turning left")
                self.world.left()
                prev_turn = "left"

            if self.show_freq != 0:
                if i % self.show_freq == 0:
                    image_data = self.world.request_image()#np.array(self.world)
                    plt.imshow(image_data)
                    plt.show()
                i += 1

            self.update_direction()

#         self.world.turn(Turn.Stop)

    @staticmethod
    def solve_triangle(theta, a):
        b = a * tan(theta)
        c = a / cos(theta)
        return b, c

    def dist_to_wall(self):
        if self.targ_dir == 0:
            if (3 * pi / 2) <= self.direction <= (2 * pi):
                a = self.world.get_y() - (self.c_targ_y - 0.5)
                theta = self.direction - (3 * pi / 2)
            else:
                a = (self.c_targ_y + 0.5) - self.world.get_y()
                theta = self.direction
        elif self.targ_dir == 90:
            if 0 <= self.direction <= (pi / 2):
                a = (self.c_targ_x + 0.5) - self.world.get_x()
                theta = self.direction
            else:
                a = self.world.get_x() - (self.c_targ_x - 0.5)
                theta = pi - self.direction
        elif self.targ_dir == 180:
            if (pi / 2) <= self.direction <= pi:
                a = (self.c_targ_y + 0.5) - self.world.get_y()
                theta = self.direction - (pi / 2)
            else:
                a = self.world.get_y() - (self.c_targ_y - 0.5)
                theta = (3 * pi / 2) - self.direction
        elif self.targ_dir == 270:
            if pi <= self.direction <= 3 * pi / 2:
                a = self.world.get_x() - (self.c_targ_x - 0.5)
                theta = self.direction - pi
            else:
                a = (self.c_targ_x + 0.5) - self.world.get_x()
                theta = (2 * pi) - self.direction

        b, c = self.solve_triangle(theta, a)

        if b < self.dist:
            return c
        else:
            return b

    def move_to_step(self):
#         self.world.turn(Turn.Stop)
        i = 0
        while (
            not in_targ_cell(
                self.base_dir, self.c_targ_x, self.c_targ_y, self.curr_x, self.curr_y
            )
            and self.step > 0.1
        ):
            angle_label = self.filename_from_angle_deg(0.0, self.img_num)
            if self.img_dir != None:
                if self.stack_dir:
                    self.world.save_png(
                        os.path.join(self.img_dir, f"{angle_label}")
                    )
                    self.img_num += 1
                else:
                    self.world.save_png(
                        os.path.join(
                            self.img_dir, "straight", f"{self.img_num_s:05}.png"
                        )
                    )
                    self.img_num_s += 1

            self.world.forward()
#             self.world.update()

            self.curr_x = self.world.get_x()
            self.curr_y = self.world.get_y()

            if self.show_freq != 0:
                if i % self.show_freq == 0:
                    image_data = self.world.request_image()#np.array(self.world)
                    plt.imshow(image_data)
                    plt.show()
                i += 1

            self.step -= self.world.rc_walk_speed
            self.update_dist()

#         self.world.walk(Walk.Stop)


class Navigator:
    def __init__(self, maze, img_dir=None):
        # establish connection to game     
        print(sys.path)
        ue4.connect(timeout=5)
        if not ue4.isconnected():
            print("UnrealCV server is not running.")
        else:
            print(ue4.request("vget /unrealcv/status"))
        self.world = UE4EnvWrapper(ue4)#PycastWorld(320, 240, maze)
        self.world.change_level(maze)
        self.img_dir = img_dir

        # getting directions
        with open(maze, "r") as in_file:
            png_count = int(in_file.readline())
            for _ in range(png_count):
                in_file.readline()

            _, dim_y = in_file.readline().split()
            for _ in range(int(dim_y)):
                in_file.readline()

            self.directions = in_file.readlines()
        
        self.num_directions = len(self.directions)
        start_angle = enws[self.directions[0].split()[2]]
        self.world.set_pose(angle = start_angle)
        
        self.angles = []

    def navigate(self, index, show_dir=False, show_freq=0):
        _, _, s_base_dir = self.directions[index].split()
        targ_x, targ_y, s_targ_dir = self.directions[index + 1].split()
        targ_x, targ_y = int(targ_x), int(targ_y)

        # convert from string
        base_dir = enws[s_base_dir]
        targ_dir = enws[s_targ_dir]

        if show_dir:
            print(f"Directions: {targ_x}, {targ_y}, {s_targ_dir}")

        # center of target cell
        c_targ_x = targ_x + 0.5
        c_targ_y = targ_y + 0.5

        driver = Driver(
            c_targ_x, c_targ_y, base_dir, targ_dir, self.world, self.img_dir, show_freq
        )

        while not in_targ_cell(
            base_dir, c_targ_x, c_targ_y, driver.curr_x, driver.curr_y
        ):
            print("Stepping through steps")
#             print(driver.curr_x)
#             print(driver.curr_y)
            driver.set_rand_angle()
#             print("out of set_rand_angle")
#             print("inside turn to angle")
            driver.turn_to_angle()
#             print("outside turn to angle")
            driver.set_rand_step()
#             print("outside set rand step")
#             print("inside move to step")
            driver.move_to_step()
#             print("out of move to step")

            self.angles.append(driver.get_angle())

    def plot_angles(self):
        for a in self.angles:
            print(a)


def main():
    maze = sys.argv[1] if len(sys.argv) > 1 else "../Mazes/maze01.txt"
    show_freq = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # frequency to show frames
    img_dir = sys.argv[3] if len(sys.argv) > 3 else None  # directory to save images to
    show_dir = (
        bool(distutils.util.strtobool(sys.argv[4])) if len(sys.argv) > 4 else False
    )

    navigator = Navigator(maze, img_dir)

    j = 0
    while j < navigator.num_directions - 1:
        navigator.navigate(j, show_dir=show_dir, show_freq=show_freq)
        j += 1

    navigator.plot_angles()


if __name__ == "__main__":
    main()
