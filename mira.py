# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
import random
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    # if (self.automaticTuning):
    #     Cgrid = [0.002, 0.004, 0.008]
    # else:
    #     Cgrid = [self.C]
    Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    "*** YOUR CODE HERE ***"
    self.initializeWeightsToZero()    # no "1" feature like perceptron this time, only image features have weights
    score = util.Counter()    # counter from labels to scores
    correctLabel = None
    guessedLabel = None

    for iteration in range(self.max_iterations):
      print("Starting MIRA iteration ", iteration, "...")
      for i in range(len(trainingData)):
        "*** YOUR CODE HERE ***"
        datum = trainingData[i]
        correctLabel = trainingLabels[i]
        for label in self.legalLabels:
          score[label] = 0
          for feature in self.features:
            score[label] += datum[feature] * self.weights[label][feature]
        guessedLabel = score.argMax()   # guess the label with best score
        if (guessedLabel == correctLabel):    # Weights work, don't touch anything! Otherwise, lower the guessed label's weights and raise the correct label's weights
          continue
        # calculate t, the variable step size
        t = min(Cgrid[0], (((self.weights[guessedLabel] - self.weights[correctLabel]) * datum + 1) / (2 * (datum * datum))))
        # adjust weights according to t
        for feature in self.weights[correctLabel]:
          self.weights[correctLabel][feature] += t * datum[feature]
          self.weights[guessedLabel][feature] -= t * datum[feature]

          
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


