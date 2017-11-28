#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import pprint

#
# Global variables
# Setup optimal string and GA input variables.
#

OPTIMAL     = "Hello, World"
DNA_SIZE    = len(OPTIMAL)
POP_SIZE    = 20
GENERATIONS = 5000

MAX_COLUMN = 5
MAX_ROW = 5
MIN_COLUMN = 2
MIN_ROW = 3

#
# Helper functions
# These are used as support, but aren't direct GA-specific functions.
#

def weighted_choice(items):
  """
  Chooses a random element from items, where items is a list of tuples in
  the form (item, weight). weight determines the probability of choosing its
  respective item. Note: this function is borrowed from ActiveState Recipes.
  """
  weight_total = sum((item[1] for item in items))
  n = random.uniform(0, weight_total)
  for item, weight in items:
    if n < weight:
      return item
    n = n - weight
  return item

def random_state(numStates, numLetters):
  """
  Return a random character between ASCII 32 and 126 (i.e. spaces, symbols,
  letters, and digits). All characters returned will be nicely printable.
  """
  next_state = random.randrange(0, numStates)
  replace_letter = random.randrange(0, numLetters)
  movement = random.randrange(0, 2)
  return { "next_state": next_state, "replace_letter": replace_letter, "movement": movement }

# TODO(Uriel96)
def random_population():
  tables = []
  for i in range(POP_SIZE):
    num_rows = random.randrange(MIN_ROW, MAX_ROW+1)
    num_columns = random.randrange(MIN_COLUMN, MAX_COLUMN+1)
    table = []
    for row in range(num_rows):
      table.append([])
      for column in range(num_columns):
        table[row].append(random_state(num_rows, num_columns))
    tables.append(table)
  return tables

def print_table(table):
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(table)

# TODO(LewisErick)
# Input: Numpy Matrix
# Output: Numpy Matrix
def get_training_set(parsed_input):
    return parsed_input[0:parsed_input.shape[0]/2, :]

# TODO(LewisErick)
# Input: Numpy Matrix
# Output: Numpy Matrix
def get_validation_set(parsed_input):
    return parsed_input[parsed_input.shape[0]/2, :]

# TODO(LewisErick)
def predict(population, training_set):
    return None

# TODO(LewisErick)
def calculate_performance(training_set, predicted_output_train):
    return None

# TODO(LewisErick)
def shrink_population(population):
    return None

# TODO(LewisErick)
def augment_population(population):
    return None

# TODO(Uriel96)
def append_generation(population):
    return None

# TODO(Uriel96)
def create_next_generation(population):
    new_population = []
    for table in population:
      table_A = pick_random_table(population)
      table_B = pick_random_table(population)
      print_table(table_A)
      print_table(table_B)
      new_table = cross_over(table_A, table_B)
      new_population.append(new_table)
    return new_population

def cross_over(table_A, table_B):
  return []

def pick_random_table(population):
  index = 0
  r = random.randrange(0, 2)
  while r > 0:
    r = r - population[index].fitness
    index += 1
  index -= 1
  return population[index]


# TODO(LewisErick)
def mutation(population):
    return None

#
# GA functions
# These make up the bulk of the actual GA algorithm.
#

def fitness(dna):
  """
  For each gene in the DNA, this function calculates the difference between
  it and the character in the same position in the OPTIMAL string. These values
  are summed and then returned.
  """
  fitness = 0
  for c in range(DNA_SIZE):
    fitness += abs(ord(dna[c]) - ord(OPTIMAL[c]))
  return fitness
"""
def mutate(dna):
  dna_out = ""
  mutation_chance = 100
  for c in range(DNA_SIZE):
    if int(random.random()*mutation_chance) == 1:
      dna_out += random_char()
    else:
      dna_out += dna[c]
  return dna_out
"""

def crossover(dna1, dna2):
  """
  Slices both dna1 and dna2 into two parts at a random index within their
  length and merges them. Both keep their initial sublist up to the crossover
  index, but their ends are swapped.
  """
  pos = int(random.random()*DNA_SIZE)
  return (dna1[:pos]+dna2[pos:], dna2[:pos]+dna1[pos:])

#
# Main driver
# Generate a population and simulate GENERATIONS generations.
#

if __name__ == "__main__":
  # Paso 1: Generar Tablas Random
  # Generate initial population. This will create a list of POP_SIZE strings,
  # each initialized to a sequence of random characters.
  population = random_population()

  for i in range(POP_SIZE):
    print_table(population[i])

  create_next_generation(population)

  '''
  generation = []

  # Parse Input
  parsed_input = io.get()

  # Training Set
  training_set = get_training_set(parsed_input)

  # Validation Set
  validation_set = get_validation_set(parsed_input)

  num_iterations = input("Indica el numero de iteraciones para el entrenamiento")

  for i in range(0, num_iterations):
      # Evaluar las cadenas del input del set de entrenamiento.
      # Output: arreglo de valores verdaderos y falsos según su aceptación
      # o rechazo.
      predicted_output_train = predict(population, training_set)

      # Using training set expected values (Y's).
      precision, recall = calculate_performance(training_set,
        predicted_output_train)

      if recall < 0.5:
          shrink_population(population)
      elif precision < 0.5:
          augment_population(population)

      # Choose the best from the population for the generation
      generation = append_generation(population)

      cross_over(population)

      mutation(population)
   '''
