# Based on iohelp.py in
# https://github.com/alanbato/proyecto-sisops/blob/master/iohelp.py

import numpy as np

def get():
    parsed_input = []
    with open("input.txt") as input_file:
        for line in input_file.readlines():
          parsed_line = line.split(" ")
          parsed_input.append([str(parsed_line[0]), int(parsed_line[1])])
    return np.matrix(parsed_input)
