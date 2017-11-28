# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:01:00 2017

@author: asus
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",
                    type=int)
args = parser.parse_args()
print(args.square**2)