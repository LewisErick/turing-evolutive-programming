# Based on iohelp.py in
# https://github.com/alanbato/proyecto-sisops/blob/master/iohelp.py

import numpy as np

import random

def get():
    parsed_input = []
    with open("input.txt") as input_file:
        for line in input_file.readlines():
          parsed_line = line.split(" ")
          parsed_input.append([str(parsed_line[0]), int(parsed_line[1])])
    return np.matrix(parsed_input)


def set_even(amount):
    with open("input.txt", "w") as output_file:
        for i in range(amount):
            new_random_string = ""
            for j in range(random.randrange(0, 20)):
                new_random_string += str(random.randrange(0, 2))
            
            if len(new_random_string) % 2 == 0:
                new_random_string += " 1\n"
            else:
                new_random_string += " 0\n"
            output_file.write(new_random_string)
        
        