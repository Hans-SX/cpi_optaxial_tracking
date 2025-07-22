"""
Created on  Jun. 12, 2025

@author: Shih-Xian
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import time
import numpy as np
# from andor3 import Andor3
from pipython import GCSDevice
from pipython import pitools

"""
Duplicated from the moving_patterns.py in acquisition_while_moving folder.
For generating some positions to be refocused to, it is corresponding to the pattern which is adopted when the data is generated.
Ex. when the data set is acquired using BigStepForward_SmallStepBack(0, 17, 6, pattern=np.array((16, -8))).generate() in acp_w_mov.py, then the same pattern should be feed in Refocused_range().bigf_smallb()[0] in tracking.py to generate a range of positions which the algorithm will refocus to those positions.
"""

class MotionPatterns(ABC):
    @abstractmethod
    def generate(self): # -> List(float):
        pass
    # def 

class BigStepForward_SmallStepBack(MotionPatterns):
    def __init__(self, start:float, end:float, steps:int, interval:float=0.5, pattern = np.array((2, -1))):
        self.start = start
        self.end = end
        self.steps = steps
        self.interval = interval
        self.pattern = pattern

        if start < 0 or start > end or end > 17:
            raise ValueError("The range of M-231.17 actuator is 0 - 17 mm.")
        elif start + (np.ceil(steps/2) * pattern[(steps - 1) % 2] + np.floor(steps/2) * pattern[steps % 2]) * interval > end or (np.ceil((steps-1)/2) * pattern[(steps - 2) % 2] + np.floor((steps-1)/2) * pattern[(steps-1) % 2]) * interval > end:
            raise ValueError("The platform will go beyond the actuator range.")
            
    
    def generate(self):
        positions = [self.start + (np.ceil(x/2) * self.pattern[(x-1) % 2] + np.floor(x/2) * self.pattern[x % 2]) * self.interval for x in range(1, self.steps + 1)]
        return positions
    
class SinusoidalForward(MotionPatterns):
    def __init__(self, start:float, end:float, steps:int, frequency:int=3, amp=3):
        # At least 60 steps would looks like a sinusodial, 90 steps would be better. 
        self.start = start
        self.end = end
        self.samples = np.linspace(start, end, steps)
        self.freq = frequency
        self.steps = steps
        self.amp = amp

        if start < 0 or start > end or end > 17:
            raise ValueError("The range of M-231.17 actuator is 0 - 17 mm.")

    def generate(self):
        positions = self.amp * np.sin(2 * np.pi * self.freq * self.samples / self.end) + self.samples
        return positions

def target_axials(x, n, stepsize):
    """
    x: the expected position
    n: 2*n + 1 will be the total examined positions
    """
    left_values = [x - i * stepsize for i in range(1, n+1)]
    right_values = [x + i * stepsize for i in range(1, n+1)]
    return left_values[::-1] + [x] + right_values

class Refocused_range():
    """
    target_axials(x, num, dis), num and dis should be put it config.py as parameters.
    They determine the range and the width of refocused positions.
    """
    def __init__(self, shift, positions=None):
        self.pos = positions
        self.shift = shift

    def step_34(self):
        expect_ref = 6.87 - np.linspace(0.25, 16.75, 34)
        try_ref_to = [target_axials(x, 10, 0.2) for x in expect_ref]
        try_shifts = [[self.shift(try_ref_to[x][y]) for y, _ in enumerate(try_ref_to[x])] for x, _ in enumerate(try_ref_to)]
        return try_shifts

    def fixed(self):
        expect_ref = 6.87 - 0.25 * np.array(range(66))
        try_ref_to = [target_axials(x, 20, 0.05) for x in expect_ref]
        try_shifts = [[self.shift(try_ref_to[x][y]) for y, _ in enumerate(try_ref_to[x])] for x, _ in enumerate(try_ref_to)]
        return try_shifts
    
    def bigf_smallb(self):
        expect_ref = []
        for i in range(len(self.pos) - 1):
            if i == 0:
                expect_ref.append(list(np.arange(0.25, self.pos[i], 0.5))[:])
            if self.pos[i] < self.pos[i+1]:
                expect_ref.append(list(np.arange(self.pos[i] + 0.25, self.pos[i+1], 0.5))[:])
            elif self.pos[i] > self.pos[i+1]:
                expect_ref.append(list(np.arange(self.pos[i] - 0.25, self.pos[i+1], -0.5))[:])
        expect_ref = [x for xs in expect_ref for x in xs]
        expect_ref = 6.87 - np.array(expect_ref)
        try_ref_to = [target_axials(x, 20, 0.05) for x in expect_ref]
        # try_ref_to = [target_axials(x, 5, 0.3) for x in expect_ref]
        try_shifts = [[self.shift(try_ref_to[x][y]) for y, _ in enumerate(try_ref_to[x])] for x, _ in enumerate(try_ref_to)]
        return try_shifts, expect_ref
    
    def sinusoidal(self):
        expect_ref = 6.87 - self.pos
        try_ref_to = [target_axials(x, 20, 0.05) for x in expect_ref]
        # try_ref_to = [target_axials(x, 5, 0.3) for x in expect_ref]
        try_shifts = [[self.shift(try_ref_to[x][y]) for y, _ in enumerate(try_ref_to[x])] for x, _ in enumerate(try_ref_to)]
        return try_shifts, expect_ref