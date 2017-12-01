#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import io
import numpy as np

import pprint

#
# Global variables
# Setup optimal string and GA input variables.
#

OPTIMAL     = "Hello, World"
DNA_SIZE    = len(OPTIMAL)
POP_SIZE    = 20
GENERATIONS = 5000

# columns -> language elements
NUM_COLUMNS = 5

# rows -> number of states (minimum 3: initial, acceptance, rejection)
NUM_ROWS = 5

MIN_COLUMN = 2
MIN_ROW = 3

MAX_COLUMN = 99
MAX_ROW = 99

MAX_AUGMENT_RATE = 3

LANGUAGE = {"0": 0, "1": 1}

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
def random_population(num_rows, num_columns):
  tables = []
  for i in range(POP_SIZE):
    table = []
    for row in range(num_columns):
      table.append([])
      for column in range(num_columns):
        table[row].append(random_state(num_rows, num_columns))
    tables.append(table)
  return tables

# TODO(Uriel96)
def print_table(table):
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(table)

# Input: Numpy Matrix
# Output: Numpy Matrix
def get_training_set(parsed_input):
    return parsed_input[0:parsed_input.shape[0]/2, :]

# Input: Numpy Matrix
# Output: Numpy Matrix
def get_validation_set(parsed_input):
    return parsed_input[parsed_input.shape[0]/2:, :]

# Maps the symbol to its index in the transition table.
def matrix_column(symbol):
    try:
        LANGUAGE[symbol]
    except KeyError:
        return 0

# Input: Population, Training Set
# Output: Matrix
#    Each row represents a table in the population
#       Each cell in the row represents the acceptance or rejection of the
#       training when run through the transition table.
def predict(population, training_set):
    training_set_x = training_set[:, 0:training_set.shape[1]-1]
    training_set_y = training_set[:, training_set.shape[1]:]

    predict_matrix = []

    #Erick, lo agregue para que no me marcara error
    basestring = ""
    for table in population:
        # Process acceptance () or denial (false) for each example in the
        # transition.
        for example in training_set_y:
            assert isinstance(example, basestring)
            # Initial state of Turing Machine
            state = 0
            # Timeout control
            timeout = 0
            # Head of Turing Machine
            head = 0
            # Example copy for modifications
            mod_example = example

            while timeout < 1000 and state is not 1 and state is not 2 and head >= len(example):
                if head < 0:
                    current_character = ""
                else:
                    current_character = mod_example[head]
                column_index = matrix_column(current_character)

                transition = table[state][column_index]
                next_state = transition["next_state"]
                replace_letter = transition["replace_letter"]
                movement = transition["movement"]

                state = next_state
                mod_example[head] = replace_letter

                if movement == 0:
                    head -= 1
                else:
                    head += 1

                timeout += 1

            # The run timed out. Invalid transition table for this example.
            if timout >= 1000:
                predict_row.append(False)
            # 1 is the row of the accepted state
            elif state is 1:
                predict_row.append(True)
            # 2 is the row of the rejected state.
            elif state is 2:
                predict_row.append(False)

        predict_matrix.append(predict_row)

    return predict_matrix

# Input: Training set, Predicted Output (matrix)
# Output: Precision List, Recall List, Accuracy List
#
# Precision = true positives / (true positivies + false positives)
# Recall = true positives / (true positives + false negatives)
# Accuracy = how many were right / number of examples
def calculate_performance(training_set, predicted_output_train):
    # Array with the precision values (0 to 1) for each of the training examples.
    precision = []
    # Array with the recall values (0 to 1) for each of the training examples.
    recall = []
    # Array with the accuracy values (0 to 1) for each of the training examples.
    accuracy = []

    training_set_y = list(training_set[:, training_set.shape[1]-1])
    for table_output in predicted_output_train:
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for prediction, real_value in zip(table_output, training_set_y):
            if prediction == real_value:
                if prediction is True:
                    true_positives += 1
                else:
                    true_negatives += 1
            if prediction != real_value:
                if prediction is True:
                    false_positives += 1
                else:
                    false_negatives += 1
        precision.append(true_positives/(true_positives+false_positives))
        recall.append(true_positives/(true_positives+false_negatives))
        accuracy.append((true_positives+true_negatives)/training_set.shape[0])

    return [precision, recall, accuracy]

def shrink_population(population):
    if (population.shape[0] > MIN_ROW):
        new_row_size = population.shape[0] - random.randrange(0, population.shape[0]-MIN_ROW)
        population = population[0:new_row_size,:]
    return population

# Adds more rows with randomly generated states to the Turing Machine transition
# table.
#
# This translates to: adding more states to the machine.
def augment_population(population):
    if (population.shape[0] < MAX_ROW):
        append_size = random.randrange(1, MAX_AUGMENT_RATE)
        population_to_append = random_population(append_size, NUM_COLUMNS)
        for append_table, table in zip(population_to_append, population):
            for row in append_table:
                table.append(row)
    return population

# Input: Population, Accuracy List for each table in the population.
def append_generation(population, accuracy=None):
  new_population = []
  for table in population:
    #Pick Two Tables
    table_A = pick_random_table(population, accuracy)
    table_B = pick_random_table(population, accuracy)

    #Cross Them
    new_table = cross_over(table_A, table_B)

    #Mutate Table
    new_table = mutation(new_table)

    #Add It to New Population
    new_population.append(new_table)
  return new_population

# TODO(Uriel96)
def cross_over(table_A, table_B):
  pos = random.randrange(1, NUM_ROWS-1)
  return random.choice([table_A[:pos] + table_B[pos:], table_B[:pos] + table_A[pos:]])

# TODO(Uriel96)
# Input: Population, Accuracy List for each table in the population.
def pick_random_table(population, accuracy):
  #TODO: Pick Random Table Based on Fitness
  '''
  index = 0
  r = random.randrange(0, 2)
  while r > 0:
    r = r - population[index].fitness
    index += 1
  index -= 1
  return population[index]
  '''
  return population[random.randrange(0, len(population))]

# Changes randomly one of the following:
# next_state, replace_letter, movement
# for all cells in the matrix.
def mutation(population):
    for i in range(0, len(population)):
        for j in range(0, len(population[0])):
            r = random.randrange(0, 3)
            new_population_state = population[i][j]
            if r > 0:
                # Next-Step
                if r == 1:
                    new_population_state["next_step"] = random.randrange(0, NUM_ROWS)
                # Replace Letter
                elif r == 2:
                    new_population_state["replace_letter"] = random.randrange(0, NUM_COLUMNS)
                # Movement
                elif r == 3:
                    new_population_state["movement"] = random.randrange(0, 2)
                population[i][j] = new_population_state

    return population

if __name__ == "__main__":
  # Parse Input
  parsed_input = io.get()

  # Paso 1: Generar Tablas Random
  # Generate initial population. This will create a list of POP_SIZE strings,
  # each initialized to a sequence of random characters.
  population = random_population(NUM_ROWS, NUM_COLUMNS)

  # x is your dataset
  np.random.shuffle(parsed_input)

  # Training Set
  training_set = get_training_set(parsed_input)

  # Validation Set
  validation_set = get_validation_set(parsed_input)

  num_iterations = input("Indica el numero de iteraciones para el entrenamiento")

  population = append_generation(population)
  '''

  for i in range(0, num_iterations):
      # Evaluar las cadenas del input del set de entrenamiento.
      # Output: arreglo de valores verdaderos y falsos según su aceptación
      # o rechazo.
      predicted_output_train = predict(population, training_set)

      # Using training set expected values (Y's).
      precision, recall, accuracy = calculate_performance(training_set,
        predicted_output_train)

      average_precision = reduce(lambda x, y: x + y, precision) / len(precision)
      average_recall = reduce(lambda x, y: x + y, recall) / len(recall)

      # Small precision means we're having too many false positives: our
      # population of transition tables is overfitting.
      if average_precision < 0.5:
          shrink_population(population)
      # Small recall means we're having too many false negatives: our population
      # of transition tables is underfitting.
      elif recall < 0.5:
          augment_population(population)

      # Choose the best from the population for the generation
      generation = append_generation(population, accuracy)
   '''
