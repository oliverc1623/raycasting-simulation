#!/usr/bin/env python

from skimage.io import imread, imsave
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
        
        self.turn_speed = 5
        self.walk_speed = 25
        
        self.set_pose()
        
    def set_pose(self, *, x=None, y=None, angle=None):
        self.x = x if x else self.x
        self.y = y if y else self.y
        self.angle = angle if angle else self.angle
        self.ue4.request(f"vset /camera/0/pose {self.x} {self.y} 100 0 {self.angle} 0")
    
    def left(self):
        self.set_pose(angle=self.angle - self.turn_speed)
        
    def right(self):
        self.set_pose(angle=self.angle + self.turn_speed)
        
    def forward(self):
        # TODO: not working (see InteractiveUnreal.py)
        rad = math.radians(self.angle)
        x_diff, y_diff = round(self.walk_speed*math.cos(rad)), round(self.walk_speed*math.sin(rad))
        self.set_pose(x=self.x + x_diff, y=self.y + y_diff)
        
    def request_image(self):
        image_data = ue4.request(f"vget /camera/0/lit png")
        return read_png(image_data)
    
    def get_dir_x(self):
        return math.cos(math.radians(self.angle))
    
    def get_dir_y(self):
        return math.sin(math.radians(self.angle))
    
    def show(self):
        plt.imshow(self.request_image())
        
    def save_png(self, filepath):
        img = self.request_image()
        imsave(filepath, img)

        