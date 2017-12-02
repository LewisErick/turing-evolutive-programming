#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np

import iohelp

import pprint

#
# Global variables
# Setup optimal string and GA input variables.
#

TIMEOUTS = 0

POP_SIZE = 100

# columns -> language elements
NUM_COLUMNS = 5

# rows -> number of states (minimum 3: initial, acceptance, rejection)
NUM_ROWS = 5

MIN_COLUMN = 2
MIN_ROW = 3

MAX_COLUMN = 99
MAX_ROW = 99

MAX_AUGMENT_RATE = 3

LANGUAGE = {"0": 0, "1": 1, " ": 2, "X": 3, "Y": 4}
LANGUAGE_INDEX = ["0", "1", " ", "X", "Y"]

INFINITE_TAPE_SIZE = 50
TIMEOUT_LIMIT = 1000

def index_to_letter(index):
    return LANGUAGE_INDEX[index]

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

def print_table(table):
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(table)

# Input: Numpy Matrix
# Output: Numpy Matrix
def get_training_set(parsed_input):
    return parsed_input[0:int(parsed_input.shape[0]/2), :]

# Input: Numpy Matrix
# Output: Numpy Matrix
def get_validation_set(parsed_input):
    return parsed_input[int(parsed_input.shape[0]/2):, :]

# Maps the symbol to its index in the transition table.
def matrix_column(symbol):
    try:
        return LANGUAGE[symbol]
    except KeyError:
        return 0

# Input: Population, Training Set
# Output: Matrix
#    Each row represents a table in the population
#       Each cell in the row represents the acceptance or rejection of the
#       training when run through the transition table.
def predict(population, training_set):
    global TIMEOUTS
    training_set_x = training_set[:, 0:training_set.shape[1]-1]
    training_set_y = training_set[:, training_set.shape[1]-1:]

    predict_matrix = []

    #Erick, lo agregue para que no me marcara error
    basestring = ""
    for table in population:
        # Process acceptance () or denial (false) for each example in the
        # transition.
        predict_row = []
        for example in training_set_x:
            # Initial state of Turing Machine
            state = 0
            # Timeout control
            timeout = 0
            # Head of Turing Machine
            head = 0
            # Example copy for modifications
            mod_example = list(str(example.item(0)))

            for i in range(0, TIMEOUT_LIMIT*2):
                mod_example.append(" ")

            while timeout < TIMEOUT_LIMIT and state is not 1 and state is not 2 and head > TIMEOUT_LIMIT*-1 and head < len(mod_example) - TIMEOUT_LIMIT:
                if head < 0:
                    current_character = ""
                else:
                    current_character = mod_example[head]
                column_index = matrix_column(str(current_character))

                if state >= len(table):
                    state = len(table)-1
                transition = table[state][column_index]

                #print("Transition: {}".format(transition))
                next_state = transition["next_state"]
                replace_letter = index_to_letter(transition["replace_letter"])
                movement = transition["movement"]

                state = next_state
                mod_example[head] = replace_letter

                #print("Movement: {}".format(movement))
                #print("State: {}".format(state))
                #print("****")

                if movement == 0:
                    head -= 1
                else:
                    head += 1

                timeout += 1

            # The run timed out. Invalid transition table for this example.
            if timeout >= TIMEOUT_LIMIT:
                TIMEOUTS += 1
                predict_row.append(False)
            # 1 is the row of the accepted state
            elif state is 1:
                predict_row.append(True)
            # 2 is the row of the rejected state.
            else:
                predict_row.append(False)

        predict_matrix.append(predict_row)
        #print("---")
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

    #print(predicted_output_train)

    training_set_y = list(training_set[:, training_set.shape[1]-1])
    for table_output in predicted_output_train:
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for prediction, real_value in zip(table_output, training_set_y):
            real_value = int(real_value.item(0))
            if real_value == 0:
                real_value = False
            elif real_value == 1:
                real_value = True
            #print(real_value)
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
        #print("True positives: {}".format(true_positives))
        #print("True negatives: {}".format(true_negatives))
        #print("False positives: {}".format(false_positives))
        #print("False negatives: {}".format(false_negatives))
        if true_positives+false_positives == 0:
            precision.append(0)
        else:
            precision.append(float(true_positives)/float(true_positives+false_positives))
        if true_positives+false_negatives == 0:
            recall.append(0)
        else:
            recall.append(float(true_positives)/float(true_positives+false_negatives))
        accuracy.append(float(true_positives+true_negatives)/float(training_set.shape[0]))

    return [precision, recall, accuracy]

def shrink_population(population):
    new_population = []
    for table in population:
        if (len(table) > MIN_ROW):
            new_row_size = len(table) - random.randrange(0, len(table)-MIN_ROW)
            table = np.matrix(table)
            table = table[0:new_row_size,:]
            table = table.tolist()
        new_population.append(table)
    return new_population

# Adds more rows with randomly generated states to the Turing Machine transition
# table.
#
# This translates to: adding more states to the machine.
def augment_population(population):
    if len(population) < MAX_ROW:
        append_size = random.randrange(1, MAX_AUGMENT_RATE)
        population_to_append = random_population(append_size, NUM_COLUMNS)
        for append_table, table in zip(population_to_append, population):
            for row in append_table:
                table.append(row)
    return population

# Input: Population, Accuracy List for each table in the population.
def append_generation(population, accuracy=None, precision=None, recall=None):
    new_population = []

    if accuracy is not None:
        average_precision = 0
        average_recall = 0
        performances = []
        for k in range(0, len(accuracy)):
            performances.append((accuracy[k], precision[k], recall[k]))
        performances.sort(reverse=True)
        for j in range(0, int(len(performances)/2)):
            average_precision = performances[j][1]
            average_recall = performances[j][2]
        average_precision = average_precision / int(len(performances)/2)
        average_recall = average_recall / int(len(performances)/2)

        #print("Average precison: {}".format(average_precision))
        #print("Average recall: {}".format(average_recall))

        if average_precision < 0.5:
            #print("Shrink")
            #population = shrink_population(population)
            population = augment_population(population)
        elif average_recall < 0.5:
            #print("Augment")
            #population = shrink_population(population)
            population = augment_population(population)

    for table in population:
      #Pick Two Tables
      table_A = pick_random_table(population, accuracy)
      table_B = pick_random_table(population, accuracy)

      #Cross Them
      new_table = cross_over(table_A, table_B)

      #Mutate Table
      new_table = mutation(new_table)
      new_population.append(new_table)
    return new_population

def cross_over(table_A, table_B):
  pos = random.randrange(1, NUM_ROWS-1)
  return random.choice([table_A[:pos] + table_B[pos:], table_B[:pos] + table_A[pos:]])

# Input: Population, Accuracy List for each table in the population.
def pick_random_table(population, accuracies):
    if accuracies is not None:
      """
      Chooses a random element from items, where items is a list of tuples in
      the form (item, weight). weight determines the probability of choosing its
      respective item. Note: this function is borrowed from ActiveState Recipes.
      """
      weight_total = sum(accuracies)
      n = random.uniform(0, weight_total)
      for table, accuracy in zip(population, accuracies):
        if n < accuracy:
          return table
        n = n - accuracy
      return table
    else:
      return population[random.randrange(0, len(population))]

# Changes randomly one of the following:
# next_state, replace_letter, movement
# for all cells in the matrix.
def mutation(table):
    for i in range(0, len(table)):
        for j in range(0, len(table[0])):
            #TODO: variar la magnitud del -1000 en base al accuracy de la tabla.
            r = random.randrange(-1000, 3)
            new_table_state = table[i][j]
            if r > 0:
                # Next-Step
                if r == 1:
                    new_table_state["next_state"] = random.randrange(0, NUM_ROWS)
                # Replace Letter
                elif r == 2:
                    new_table_state["replace_letter"] = random.randrange(0, NUM_COLUMNS)
                # Movement
                elif r == 3:
                    new_table_state["movement"] = random.randrange(0, 2)
                table[i][j] = new_table_state

    return table

if __name__ == "__main__":
    # Parse Input
    parsed_input = iohelp.get()

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

    num_iterations = int(input("Indica el numero de iteraciones para el entrenamiento"))

    predicted_output_train = None
    precision = None
    recall = None
    accuracy = None

    population = append_generation(population)

    for i in range(0, num_iterations):
        # Evaluar las cadenas del input del set de entrenamiento.
        # Output: arreglo de valores verdaderos y falsos según su aceptación
        # o rechazo.
        predicted_output_train = predict(population, training_set)

        #print(str(predicted_output_train) + "\n")

        # Using training set expected values (Y's).
        precision, recall, accuracy = calculate_performance(training_set,
            predicted_output_train)

        population = append_generation(population, accuracy, precision, recall)

    predicted_output_train = predict(population, training_set)

    precision, recall, accuracy = calculate_performance(training_set,
        predicted_output_train)

    print("Final Results for Training Set")

    average_precision = 0
    average_recall = 0
    performances = []
    index = 0
    for k in range(0, len(accuracy)):
        performances.append((accuracy[k], precision[k], recall[k], index))
        index += 1
    performances.sort(reverse=True)

    print("Best Accuracy: {}".format(performances[0][0]))
    print("Best Precision: {}".format(performances[0][1]))
    print("Best Recall: {}".format(performances[0][2]))
    print("Prediction Set Y: {}".format(predicted_output_train[performances[0][3]]))
    print("Training Set Y: {}".format(training_set[:,1:].tolist()))
    print(training_set.shape)

    predicted_output_validation = predict(population, validation_set)

    precision, recall, accuracy = calculate_performance(validation_set,
        predicted_output_validation)

    print("Final Results for Validation Set")

    average_precision = 0
    average_recall = 0
    performances = []
    index = 0
    for k in range(0, len(accuracy)):
        performances.append((accuracy[k], precision[k], recall[k], index))
        index += 1
    performances.sort(reverse=True)

    print("Best Accuracy: {}".format(performances[0][0]))
    print("Best Precision: {}".format(performances[0][1]))
    print("Best Recall: {}".format(performances[0][2]))
    print("Prediction Set Y: {}".format(predicted_output_validation[performances[0][3]]))
    print("Validation Set Y: {}".format(validation_set[:,1:].tolist()))
    print(validation_set.shape)

# Agregar columnas
# Quitar columnas
# Elitismo (pasar la mejor a la siguiente generación con la mejor tabla)
# Local search (ir agregando/quitando filas y evaluando accuracy)
