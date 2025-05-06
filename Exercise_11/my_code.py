#!/usr/bin/env python
import os
import re
import numpy as np
import time
import copy
import sys

import argparse


parser = argparse.ArgumentParser(
    description='sum all the numbers in a file')

parser.add_argument(
    'file_name',
    metavar='FILENAME',
    help='file containing the data.')

args = parser.parse_args()

total_sum = 0
# Open the file in read mode
with open(args.file_name, 'r') as file:
    # Read the content of the file
    content = file.read()
    # Find all numbers in the content using a regular expression
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    # Convert these numbers to floats if they have decimals, else integers, and sum them up
    total_sum = sum(float(num)  for num in numbers)
with open('sum.out', 'w') as file:
    file.write(f"The sum of all numbers in the file is: {total_sum}\n") 




