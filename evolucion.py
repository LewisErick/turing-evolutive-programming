#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Luis Erick Zul Rabasa A00818663
# Uriel Salazar Urquidi

import random
import numpy as np

import platform

import iohelp
import os

import pprint

IN_DEBUG_MODE = False

TIMEOUTS = 0

POP_SIZE = 100

NUM_COLUMNS = 5
NUM_ROWS = 5

MIN_COLUMN = 3
MIN_ROW = 3

MAX_COLUMN = 99
MAX_ROW = 99

MAX_AUGMENT_RATE = 50

LANGUAGE = {"0": 0, "1": 1, " ": 2}
LANGUAGE_INDEX = ["0", "1", " "]

INFINITE_TAPE_SIZE = 50
TIMEOUT_LIMIT = 1000

ELITISM_TOLERANCE = 0

def index_to_letter(index):
    if index > 2:
        return chr(97 + index-3)
    return LANGUAGE_INDEX[index]

#
# Helper functions
# These are used as support, but aren't direct GA-specific functions.
#

def random_state(numStates, numLetters):
  """
  Return a random character between ASCII 32 and 126 (i.e. spaces, symbols,
  letters, and digits). All characters returned will be nicely printable.
  """
  next_state = random.randrange(0, numStates)
  replace_letter = random.randrange(0, numLetters)
  movement = random.randrange(0, 3)
  return { "next_state": next_state, "replace_letter": replace_letter, "movement": movement }

def generate_random_population(num_rows, num_columns):
  tables = []
  for i in range(POP_SIZE):
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
        if symbol != "0" and symbol != "1" and symbol != " " and symbol != "":
            return ord(symbol) - ord("a")
        if symbol == "":
            return 2
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
    global NUM_COLUMNS
    training_set_x = training_set[:, 0:training_set.shape[1]-1]
    training_set_y = training_set[:, training_set.shape[1]-1:]

    predict_matrix = []

    #Erick, lo agregue para que no me marcara error
    basestring = ""
    for table in population:
        #print_table(table)
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
                if column_index >= NUM_COLUMNS:
                    column_index = random.randrange(0, NUM_COLUMNS)

                #print(len(table))
                #print(len(table[0]))
                #print(state)
                #print(column_index)
                if table[state][column_index]["next_state"] >= len(table):
                    if len(table) > 4:
                        table[state][column_index]["next_state"] = random.randrange(3, len(table)-1)
                    elif len(table) == 4:
                        table[state][column_index]["next_state"] = 3
                    else:
                        table[state][column_index]["next_state"] = 0

                if table[state][column_index]["replace_letter"] >= len(table[0]):
                    table[state][column_index]["replace_letter"] = random.randrange(0, len(table[0])-1)


                #print(current_character)
                transition = table[state][column_index]
                #print(transition)
                next_state = transition["next_state"]
                replace_letter = index_to_letter(transition["replace_letter"])
                movement = transition["movement"]

                state = next_state
                mod_example[head] = replace_letter

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
            real_value = int(real_value.item(0))
            if real_value == 0:
                real_value = False
            elif real_value == 1:
                real_value = True
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
        if true_positives+false_positives == 0:
            precision.append(0)
        else:
            precision.append(float(true_positives)/float(true_positives+false_positives))
        if true_positives+false_negatives == 0:
            recall.append(0)
        else:
            recall.append(float(true_positives)/float(true_positives+false_negatives))
        accuracy.append(float(true_positives+true_negatives)/float(training_set.shape[0]))

    return [accuracy, precision, recall]

def shrink_population(population, accuracy):
    new_population = []
    if len(population[0]) > MIN_ROW:
        max_augment = int(MAX_AUGMENT_RATE*(1-accuracy))+1
        new_row_size = len(population[0]) - max_augment
        if new_row_size < MIN_ROW:
            new_row_size = MIN_ROW
        new_column_size = len(population[0][0]) - max_augment
        if new_column_size < MIN_COLUMN:
            new_column_size = MIN_COLUMN
        for table in population:
            new_table = table[0:new_row_size]
            index = 0
            for row in new_table:
                new_table[index] = row[0:new_column_size]
                index += 1
            new_table[0] = [random_state(new_row_size, new_column_size) for i in range(0, new_column_size)]
            new_population.append(new_table)
        return new_population
    return population


# Adds more rows with randomly generated states to the Turing Machine transition
# table.
#
# This translates to: adding more states to the machine.
def augment_population(population, accuracy):
    new_population = []
    if len(population[0]) < MAX_ROW:
        max_augment = int(MAX_AUGMENT_RATE*(1-accuracy))+1
        if max_augment == 1:
            return population
        append_size = random.randrange(1, max_augment)
        num_rows = NUM_ROWS + append_size
        if num_rows > MAX_ROW:
            num_rows = MAX_ROW
        append_size_columns = random.randrange(1, max_augment)
        prev_num_columns = NUM_COLUMNS
        num_columns = NUM_COLUMNS + append_size_columns
        if num_columns > MAX_COLUMN:
            num_columns = MAX_COLUMN
        population_to_append = generate_random_population(num_rows-len(population[0]), num_columns)
        for append_table, table in zip(population_to_append, population):
            new_table = table[:]
            index = 0
            for row in new_table:
                new_row = row[:]
                for i in range(0, num_columns-len(row)):
                    new_state = random_state(num_rows, num_columns)
                    new_row.append(new_state)
                new_table[index] = new_row
                index += 1
            for row in append_table:
                new_table.append(row)
            new_table[0] = [random_state(num_rows, num_columns) for i in range(0, num_columns)]
            new_population.append(new_table)
        return new_population
    return population

def shrink_population_if_needed(population, accuracy, precision, recall, training_set):
    if IN_DEBUG_MODE:
        print("Verifying if shrinking can improve accuracy...")
    shrinked_population = shrink_population(population, max(accuracy))
    predicted_output_shrink = predict(shrinked_population, training_set)
    shrink_accuracies, shrink_precisions, shrink_recall = calculate_performance(training_set,
        predicted_output_shrink)

    shrink_timeout = 0
    dimensions_tried = []

    while max(shrink_accuracies) < max(accuracy) and shrink_timeout < TIMEOUT_LIMIT:
        shrink_timeout += 1
        shrinked_population = shrink_population(population, max(accuracy))
        if (len(shrinked_population[0]), len(shrinked_population[0][0])) in dimensions_tried:
            continue
        dimensions_tried.append((len(shrinked_population[0]), len(shrinked_population[0][0])))
        if (len(shrinked_population[0]), len(shrinked_population[0][0])) == (len(population[0]), len(population[0][0])):
            continue
        for table in shrinked_population:
            table = mutation(table)
        predicted_output_shrink = predict(shrinked_population, training_set)
        shrink_accuracies, shrink_precisions, shrink_recall = calculate_performance(training_set,
            predicted_output_shrink)

    if max(shrink_accuracies) > max(accuracy):
        if IN_DEBUG_MODE:
            print("Shrinking improved predictions. New {} Prev {}".format(max(shrink_accuracies),
                max(accuracy)))
        shrinked = True
        population = shrinked_population
        accuracy = shrink_accuracies
        precision = shrink_precisions
        recall = shrink_recall
        NUM_ROWS = len(population[0])
        NUM_COLUMNS = len(population[0][0])
    else:
        if IN_DEBUG_MODE:
            print("Shrinking didn't improve predictions.")
    return population, accuracy, precision, recall

def augment_population_if_needed(population, accuracy, precision, recall, training_set):
    if IN_DEBUG_MODE:
        print("Verifying if augmenting can improve accuracy...")
    augmented_population = augment_population(population, max(accuracy))
    predicted_output_augment = predict(augmented_population, training_set)
    augment_accuracies, augment_precisions, augment_recall = calculate_performance(training_set,
        predicted_output_augment)

    augment_timeout = 0
    while max(augment_accuracies) < max(accuracy) and augment_timeout < TIMEOUT_LIMIT:
        augment_timeout += 1
        augmented_population = augment_population(population, max(accuracy))
        for table in augmented_population:
            table = mutation(table)
        predicted_output_augment = predict(augmented_population, training_set)
        augment_accuracies, augment_precisions, augment_recall = calculate_performance(training_set,
            predicted_output_augment)

    if max(augment_accuracies) > max(accuracy):
        if IN_DEBUG_MODE:
            print("Augmentation improved predictions. New {} Prev {}".format(max(augment_accuracies),
                max(accuracy)))
        population = augmented_population
        accuracy = augment_accuracies
        precision = augment_precisions
        recall = augment_recall
        NUM_ROWS = len(population[0])
        NUM_COLUMNS = len(population[0][0])
    else:
        if IN_DEBUG_MODE:
            print("Augmenting didn't improve predictions.")
    return population, accuracy, precision, recall

# Input: Population, Accuracy List for each table in the population.
def create_next_generation(population, accuracy, precision, recall, training_set):
    population, accuracy, precision, recall = shrink_population_if_needed(population,
        accuracy, precision, recall, training_set)

    population, accuracy, precision, recall = augment_population_if_needed(population,
        accuracy, precision, recall, training_set)

    new_population = []
    for i in range(0,int(POP_SIZE/2)):
      #Pick Two Tables
      table_A, accuracy_A = pick_random_table(population, accuracy)
      table_B, accuracy_B = pick_random_table(population, accuracy)

      new_table_A = None
      new_table_B = None
      #Cross Them
      if accuracy_A is not None and accuracy_B is not None:
          new_table_A, new_table_B = cross_over(table_A, table_B, (accuracy_A+accuracy_B)/2.0)
      else:
          new_table_A, new_table_B = cross_over(table_A, table_B)

      #Mutate Table
      if accuracy_A is not None and accuracy_B is not None:
          new_table_A = mutation(new_table_A, (accuracy_A+accuracy_B)/2.0)
          new_table_B = mutation(new_table_B, (accuracy_A+accuracy_B)/2.0)
      else:
          new_table_A = mutation(new_table_A)
          new_table_B = mutation(new_table_B)
      new_population.append(new_table_A)
      new_population.append(new_table_B)
    return new_population, True

def cross_over(table_A, table_B, accuracy=0):
    pos = random.randrange(1, len(table_A)-1)
    return ([table_A[:pos] + table_B[pos:], table_B[:pos] + table_A[pos:]])

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
      table = None
      for table, accuracy in zip(population, accuracies):
        if n < accuracy:
          return table, accuracy
        n = n - accuracy
      return table
    rand_index = random.randrange(0, len(population))
    return population[rand_index], None

# Changes randomly one of the following:
# next_state, replace_letter, movement
# for all cells in the matrix.
def mutation(table, accuracy=0):
    for i in range(0, len(table)):
        for j in range(0, len(table[i])):
            r = random.randrange(int(-100*accuracy), 3)
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
                    new_table_state["movement"] = random.randrange(0, 3)
                table[i][j] = new_table_state

    return table

def get_best_performance(accuracies, precisions, recalls):
    average_precision = 0
    average_recall = 0
    performances = []
    index = 0
    for i in range(0, len(accuracies)):
        performances.append((accuracies[i], precisions[i], recalls[i], index))
        index += 1
    performances.sort(reverse=True)
    return performances[0]

def clear_terminal():
    if platform.system() == 'Windows':
        os.system( 'cls' )
    else:
        os.system( 'clear' )

if __name__ == "__main__":
    iohelp.set_even(100)

    # Parse Input
    parsed_input = iohelp.get()

    # x is your dataset
    np.random.shuffle(parsed_input)

    # Training Set
    training_set = get_training_set(parsed_input)

    # Validation Set
    validation_set = get_validation_set(parsed_input)

    num_generations = int(input("Number of generations: "))
    POP_SIZE = int(input("Population size: "))

    predicted_output_train = None
    precision = None
    recall = None
    accuracy = None

    # Generates a population of random tables.
    population = generate_random_population(NUM_ROWS, NUM_COLUMNS)

    for i in range(0, num_generations):
        # Evaluate the population with all the strings of the training set.
        # Output: array with accepted (as true) and rejected (as false) values.
        predicted_output_train = predict(population, training_set)

        # Get the performance for all tables.
        accuracies, precisions, recalls = calculate_performance(training_set,
            predicted_output_train)

        # Create the next generation based on the performance of the current generation.
        population, continue_creating = create_next_generation(population, accuracies, precisions, recalls,
            training_set)

        # Get the best table of all the population.
        best_accuracy, best_precision, best_recall, index = get_best_performance(accuracies, precisions, recalls)

        if best_accuracy >= 0.7:
            ELITISM_TOLERANCE += 1
        else:
            ELITISM_TOLERANCE = 0
        if ELITISM_TOLERANCE == int(num_generations/10):
            break

        if IN_DEBUG_MODE == False:
            clear_terminal()
        print("Generation #{}".format(i+1))
        print("Best Accuracy: {}".format(best_accuracy))
        print("Dimensions of table {}x{}".format(len(population[index]), len(population[index][0])))

    # Get the best table of all the generations.
    best_accuracy, best_precision, best_recall, index = get_best_performance(accuracies, precisions, recalls)

    if IN_DEBUG_MODE == False:
        clear_terminal()
    print("Best table: ")
    print_table(population[index])

    print()
    print("Final Results for Training Set")
    print("Best Accuracy: {}".format(best_accuracy))
    print("Best Precision: {}".format(best_precision))
    print("Best Recall: {}".format(best_recall))

    if IN_DEBUG_MODE:
        print("Total Timeouts {}".format(TIMEOUTS))


    # Evaluate the population of the final generation with all the string of the validation set.
    predicted_output_validation = predict(population, validation_set)

    # Get the performance for all tables of the final generation.
    validation_accuracies, validation_precisions, validation_recalls = calculate_performance(validation_set,
        predicted_output_validation)

    best_accuracy, best_precision, best_recall, index = get_best_performance(validation_accuracies, validation_precisions, validation_recalls)

    print()
    print("Final Results for Validation Set")
    print("Best Accuracy: {}".format(best_accuracy))
    print("Best Precision: {}".format(best_precision))
    print("Best Recall: {}".format(best_recall))
    print("Prediction Set Y: {}".format(predicted_output_validation[index]))
    print("Validation Set Real Y: {}".format(validation_set[:,1:].tolist()))
