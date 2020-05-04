# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import random
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels)
    self.weights == weights
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

    # initialize weights to random values
    random.seed()
    for label in self.legalLabels:
      for feature in self.features:
        self.weights[label][feature] = random.uniform(-0.001, 0.001)

    w0 = util.Counter()   # weight for the "1"-feature for each label; w0[y] = "1"-feature weight for each label 'y'
    score = util.Counter()    # counter from labels to scores
    correctLabel = None
    guessedLabel = None

    for iteration in range(self.max_iterations):
      print("Starting Perceptron iteration ", iteration, "...")
      for i in range(len(trainingData)):
          "*** YOUR CODE HERE ***"
          datum = trainingData[i]
          correctLabel = trainingLabels[i]
          for label in self.legalLabels:
            score[label] = w0[label]
            for feature in self.features:
              score[label] += datum[feature] * self.weights[label][feature]
          guessedLabel = score.argMax()   # return best scoring label
          if (guessedLabel == correctLabel):    # Weights work, don't touch anything! Otherwise, lower the guessed label's weights and raise the correct label's weights
            continue
          self.weights[correctLabel] += datum
          w0[correctLabel] += 1.0
          self.weights[guessedLabel] -= datum
          w0[guessedLabel] -= 1.0

          

    
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

