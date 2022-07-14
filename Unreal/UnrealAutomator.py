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
sys.path.append("/home/eoca2018/unrealcv/client/python") # adjust according to your unrealcv directory
# from unrealcv import client as ue4
import unrealcv
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
    """
    Determines if agent is in target cell (next cell).
    Parameters:
        base_dir (int): current direction e.g. 180, 90, ...
        c_targ_x (float): x coordinate of the next cell
        c_targ_y (float): y coordinate of the next cell
        x (float): current x coordinate
        y (float): current y coordinate
    """
    if base_dir == 0 or base_dir == 180:
        if abs(c_targ_x - x) < 0.4:
            return True
    else:
        if abs(c_targ_y - y) < 0.4:
            return True
    return False


class Driver:
    """
    This class defines a Driver object to communicate directions to the agent.
    
    Attributes: 
        c_targ_x (float): x coordinate of the next cell
        c_targ_y (float): y coordinate of the next cell
        base_dir (int): current direction e.g. 180, 90, ...
        targ_dir (int): next direction e.g. 180, 90, ...
        world (UE4EnvWrapper): Unreal Engine client - this you can modify according to UE5
        img_dir (string): directory to store image files
        show_freq (int): frequency we want to save images at to make a movie/gif. E.g. 5 means save every 5th images to make a movie
    """
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
        """
        Calculate Euclidean distance between current coordinates and target coordinates
        """
        self.dist = math.sqrt(
            (self.c_targ_x - self.world.get_x()) ** 2
            + (self.c_targ_y - self.world.get_y()) ** 2
        )

    def update_direction(self):
        """
        Updates current angle of the agent - I think
        
        self.world.get_dir_x() gets cos(theta)
        self.world.get_dir_y() gets sin(theta)
        """
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
        """
        Add a value to the target coordinates to induce more variability.
        This is suppose to help train the model to handle new mazes it has
        never seen before
        
        Parameters: 
            delta (float): value to offset the target x,y coordinates
        """
        if self.base_dir == 0 or self.base_dir == 180:
            if self.targ_dir == 90:
                return self.c_targ_x, self.c_targ_y + delta
            elif self.targ_dir == 270:
                return self.c_targ_x, self.c_targ_y - delta
        elif self.base_dir == 90 or self.base_dir == 270:
            if self.targ_dir == 0:
                return self.c_targ_x + delta, self.c_targ_y
            elif self.targ_dir == 180:
                return self.c_targ_x - delta, self.c_targ_y
        return self.c_targ_x, self.c_targ_y

    def get_angle(self):
        """
        Returns the currently angle of the agent
        """
        mod_x, mod_y = self.modified_targ(0.15) 
        # case where target position is up to the right
        # going top right -> top left
        if self.curr_x <= mod_x and self.curr_y <= mod_y:
            if mod_x == self.curr_x:
                theta = pi / 2
            else:
                theta = (atan((mod_y - self.curr_y) / (mod_x - self.curr_x))) % (2 * pi) 

        # case where target pos is up and to the left
        elif self.curr_x > mod_x and self.curr_y <= mod_y:
            if mod_y == self.curr_y:
                theta = pi
            else:
                theta = (atan((self.curr_x - mod_x) / (mod_y - self.curr_y))) % (
                    2 * pi
                ) + pi / 2

        # case where target pos is down and to the left
        elif self.curr_x > mod_x and self.curr_y > mod_y:
            if mod_x == self.curr_x:
                theta = 0
            else:
                theta = (atan((self.curr_y - mod_y) / (self.curr_x - mod_x))) % (
                    2 * pi
                ) + pi

        # case where target pos is down and to the right
        else:
            if self.curr_y == mod_y:
                theta = 0
            else:
                theta = (atan((mod_x - self.curr_x) / (self.curr_y - mod_y))) % (
                    2 * pi
                ) + 3 * pi / 2
        return theta

    def set_rand_angle(self):
        """
        Randomly select a turn angle from a normal distribution centered at the 
        current angle. This is to add a little randomness so that the model
        can traverse a new maze better. 
        
        This could be a potential spot to modify
        to set your own desired turning angles.
        """
        theta = self.get_angle()
        self.angle = rng.normal(loc=theta, scale=rand_angle_scale) % (2 * pi)

    def set_rand_step(self):
        """
        Randomly select a distance to travel in a straight line 
        from a uniform distribution 
        
        In other words select from a Uniform(0.4, distance to wall) distribution.
        """
        self.step = rng.uniform(rand_step_scale, self.dist_to_wall())

    def abs_angle_diff(self, angle):
        """
        Calculate absolute angle difference between current angle and given angle
        
        Parameters: 
            angle (float): angle - not necessarily the agent's own angle
        """
        abs_diff = abs(self.direction - angle)
        return abs_diff % (2 * pi)

    def turn_right(self, angle):
        adj_dir = self.direction #(pi - self.direction) might need this is you don't use UE coordinate system
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
        """
        Tell the agent how much it should turn based on the current angle head and target cell
        
        The amount to turn by is calculated by computing the angle difference between the current
        angle and target angle/direction
        
        Another potential place to modify to specify your own turn angles instead
        """
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
                    # You can comment out save_png if you want to test the agent faster
                    # without saving images
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
                self.world.left()
                prev_turn = "left"
            
            # This chunk saves images for every show_freq amount for the movie gif
            if self.show_freq != 0:
                if i % self.show_freq == 0:
                    image_data = self.world.request_image()#np.array(self.world)
                    plt.imshow(image_data)
                    plt.show()
                i += 1

            self.update_direction()


    @staticmethod
    def solve_triangle(theta, a):
        b = a * tan(theta)
        c = a / cos(theta)
        return b, c

    def dist_to_wall(self):
        """
        Computes distance to wall from the agent's 
        current angle to the target direction (90, 180, 270, or 0)
        """
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
        """
        Tell the agent how much to move forward by 
        """
        i = 0
        while (
            not in_targ_cell(
                self.base_dir, self.c_targ_x, self.c_targ_y, self.curr_x, self.curr_y
            )
            and self.step > 0.1
        ):
            angle_label = self.filename_from_angle_deg(0.0, self.img_num)
            if self.img_dir != None:
                # You can comment out save_png if you want to test the agent faster
                # without saving images
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

            # update current x,y
            self.curr_x = self.world.get_x()
            self.curr_y = self.world.get_y()

            # This chunk saves images for every show_freq amount for the movie gif
            if self.show_freq != 0:
                if i % self.show_freq == 0:
                    image_data = self.world.request_image()#np.array(self.world)
                    plt.imshow(image_data)
                    plt.show()
                i += 1

            self.step -= self.world.rc_walk_speed
            self.update_dist()


class Navigator:
    """
    This is a class to define an object that handles navigation inputs and the UE client
    
    Attributes: 
        maze (file): 2D txt representation of the maze
        img_dir (dir): directory to save images in
    """
    def __init__(self, maze, img_dir=None):
        # Establish connection to game     
        # Enter UnrealCV connection 
        client = unrealcv.Client(("localhost", 8999), None)
        client.connect(timeout=5)
        if not client.isconnected():
            print("UnrealCV server is not running.")
        else:
            print(client.request("vget /unrealcv/status"))
        self.world = UE4EnvWrapper(client)
        self.world.change_level(maze)
        self.img_dir = img_dir

        # getting directions
        # read the 2D maze file
        with open(maze, "r") as in_file:
            png_count = int(in_file.readline())
            for _ in range(png_count):
                in_file.readline()

            _, dim_y = in_file.readline().split()
            for _ in range(int(dim_y)):
                in_file.readline()

            self.directions = in_file.readlines()
        
        self.num_directions = len(self.directions)
        # map direction to ENWS value
        # {"Dir.EAST": 180, "Dir.NORTH": 90, "Dir.WEST": 0, "Dir.SOUTH": 270}
        start_angle = enws[self.directions[0].split()[2]]
        self.world.set_pose(angle = start_angle)
        
        self.angles = []

    def navigate(self, index, show_dir=False, show_freq=0):
        """
        Navigate tells the driver to set a randle angle, turn to it, 
        set a random step foward, and then move towards that step  
        """
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
            driver.set_rand_angle()
            driver.turn_to_angle()
            driver.set_rand_step()
            driver.move_to_step()

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
