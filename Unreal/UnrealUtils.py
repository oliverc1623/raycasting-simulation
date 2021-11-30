#!/usr/bin/env python

# from skimage.io import imread, imsave
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
import math

import sys

# sys.path.append("/home/eoca2018/unrealcv/client/python")
from unrealcv import client as ue4  # type: ignore
from unrealcv.util import read_png  # type: ignore


class UE4EnvWrapper():
    
    def __init__(self, ue4):
        self.ue4 = ue4
        
        self.x = 700
        self.y = -700
        self.angle = 0
        
        self.rc_x = 1.5
        self.rc_y = 1.5
        
        self.turn_speed = 2.5#2.5 * (3.1415926 / 180)
        self.walk_speed = 5#23.3
        self.rc_walk_speed = 0.05
        
        self.set_pose()
        
    def set_pose(self, *, x=None, y=None, angle=None):
        self.x = x if x else self.x
        self.y = y if y else self.y
        self.angle = angle if angle else self.angle
        self.ue4.request(f"vset /camera/0/pose {self.x} {self.y} 100 0 {self.angle} 0")
        
    def set_rc_pose(self, x=None, y=None, angle=None):
        self.rc_x = x if x else self.rc_x
        self.rc_y = y if y else self.rc_y
    
    def left(self):
        self.set_pose(angle=self.angle - self.turn_speed)
        
    def right(self):
        self.set_pose(angle=self.angle + self.turn_speed)
        
    def forward(self):
        rad = math.radians(self.angle)
        x_diff, y_diff = round(self.walk_speed*math.cos(rad)), round(self.walk_speed*math.sin(rad))
        delta_y = self.y + y_diff
        delta_x = self.x + x_diff
        if delta_y == 0:
            delta_y = delta_y + 1
        if delta_x == 0:
            delta_x = delta_x + 1
        self.set_pose(x=delta_x, y=delta_y)
        
        corrected_angle = self.angle
        if self.angle > 0 and self.angle < 180:
            corrected_angle = 180 - self.angle
        elif self.angle >= 180:
            corrected_angle = (180-self.angle)+360
        corrected_angle = math.radians(corrected_angle)
        x_rc_diff = round(self.rc_walk_speed*math.cos(corrected_angle), 2)
        y_rc_diff = round(self.rc_walk_speed*math.sin(corrected_angle), 2)
        self.set_rc_pose(x=self.rc_x + x_rc_diff, y=self.rc_y + y_rc_diff)
        
        
    def request_image(self):
        image_data = ue4.request(f"vget /camera/0/lit png")
        return read_png(image_data)
    
    def get_dir_x(self):
        if self.angle <= 0:
            raise ValueError('LESS THAN ZERO')
        print(f"self.angle: {self.angle}")
        corrected_angle = self.angle
        if self.angle > 0 and self.angle < 180:
            corrected_angle = 180 - self.angle
        elif self.angle >= 180:
            corrected_angle = (180-self.angle)+360
        print(f"get_dir_x: corrected angle: {corrected_angle}")
        return math.cos(math.radians(corrected_angle))
    
    def get_dir_y(self):
        if self.angle <= 0:
            raise ValueError('LESS THAN ZERO')
        corrected_angle = self.angle
        if self.angle > 0 and self.angle < 180:
            corrected_angle = 180 - self.angle
        elif self.angle >= 180:
            corrected_angle = (180-self.angle)+360
        print(f"get_dir_y: corrected angle: {corrected_angle}")
        return math.sin(math.radians(corrected_angle))
    
    def show(self):
        plt.imshow(self.request_image())
        
    def save_png(self, filepath):
        img = self.request_image()
        plt.imsave(filepath, img)
        
    def get_x(self):
        return self.rc_x
        
    def get_y(self):
        return self.rc_y
    
    def is_done(self):
        return (self.rc_x == 15.5 and self.rc_y == 15.5)
    
    # TODO: add function to change maze level in game
    def change_level(self, maze):       
        maze_name = maze.split("/")[3][:-4]
        self.ue4.request(f"vset /action/game/level {maze_name}")
        
    def get_angle(self):
        corrected_angle = self.angle
        if self.angle >= 0 and self.angle <= 180:
            corrected_angle = 180 - self.angle
        elif self.angle > 180:
            corrected_angle = (180-self.angle)+360
        return corrected_angle