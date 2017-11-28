import random

#
# Global variables
# Setup optimal string and GA input variables.
#

OPTIMAL     = "Hello, World"
DNA_SIZE    = len(OPTIMAL)
POP_SIZE    = 20
GENERATIONS = 5000

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
  nextState = random.randrange(0, numStates)
  replaceLetter = random.randrange(0, numLetters)
  movement = random.randrange(0, 2)
  return { "nextState": nextState, "replaceLetter": replaceLetter, "movement": movement }

# TODO(Uriel96)
def random_population():
  """
  Return a list of POP_SIZE individuals, each randomly generated via iterating
  DNA_SIZE times to generate a string of random characters with random_char().
  """
  max_column = 3
  max_row = 3

  tables = []
  for i in range(POP_SIZE):
    table = []
    for row in range(max_row):
      table.append([])
      for column in range(max_column):
        table[row].append(random_state(max_row, max_column))
    tables.append(table)
  return tables

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
def cross_over(population):
    return None

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
    for row in range(3):
      for column in range(3):
        print(population[i][row][column])


  generation = []

  # Parse Input
  parsed_input = io.get()

  # Training Set
  training_set = get_training_set(parsed_input)

  # Validation Set
  validation_set = get_validation_set(parsed_input)

  num_iterations = input("Indica el número de iteraciones para el entrenamiento")

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
