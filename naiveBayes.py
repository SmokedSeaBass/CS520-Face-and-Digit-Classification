# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"

    # Get prior distribution
    # P(y = true) for each label;  P(y = false) is just 1 - P(y = true)
    # priorDistrib[label]
    print("Calculating prior distributions...")
    labelTotal = len(trainingLabels) * 1.0
    labelCounts = util.Counter()
    for label in trainingLabels:
      labelCounts[label] += 1.0
    priorDistrib = util.Counter()
    for label in self.legalLabels:
      priorDistrib[label] = labelCounts[label]/labelTotal
    
    self.priorDistrib = priorDistrib

    # Get count of feature=value for each label (used for computing conditional probabilities later)
    # grid[label][(feature, value)] = count
    print("Getting feature=value counts for each label...")
    featureValueGrid = util.Counter()         # counter from labels to features=value pairs
    for label in self.legalLabels:
      featureValueGrid[label] = util.Counter()    # counter from features=value pairs to counts (of that pair)
    for i in range(len(trainingData)):
      datum = trainingData[i]
      label = trainingLabels[i]
      for feature in self.features:
        value = datum[feature]
        featureValueGrid[label][(feature, value)] += 1.0

    # Get conditional probabilities
    # p(φ_i(x) = v | y = true) and p(φ_i(x) = v | y = false)
    # probDistribTrue[label][(feature, value)]
    print("Calculating conditional probabilities...")
    probDistribTrue = util.Counter()    # list of p(φ_i(x) = v | y = true) for each value v of each feature i, for each label y 
    probDistribFalse = util.Counter()   # list of p(φ_i(x) = v | y = false) for each value v of each feature i, for each label y
    for label in self.legalLabels:
      probDistribTrue[label] = util.Counter()    # list of features=value pairs where label is true
      probDistribFalse[label] = util.Counter()    # list of features=value pairs where label is false
      for pair in featureValueGrid[label]:
        probDistribTrue[label][pair] = featureValueGrid[label][pair] / labelCounts[label]
        for label2 in self.legalLabels:
          if (label2 != label):
            probDistribFalse[label][pair] += featureValueGrid[label2][pair]
        probDistribFalse[label][pair] /= (labelTotal - labelCounts[label])
    self.probDistribTrue = probDistribTrue
    self.probDistribFalse = probDistribFalse
    return



  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    L = util.Counter()
    for datum in testData:
      for label in self.legalLabels:
        L[label] = 1        # intialize at 1, will be the L product for each label
      for feature in datum:
        value = datum[feature]
        for label in self.legalLabels:
          # L = ()
          A1 = max(0.00000000001, self.probDistribTrue[label][(feature, value)]) * 1.0
          A2 = max(0.00000000001, self.priorDistrib[label]) * 1.0
          B1 = max(0.00000000001, self.probDistribFalse[label][(feature, value)]) * 1.0
          B2 = max(0.00000000001, (1.0 - self.priorDistrib[label])) * 1.0
          # print (str(A1) + " * " + str (A2) + " / " + str(B1) + " * " + str(B2))
          L[label] *= ((A1 * A2) / (B1 * B2))
      guesses.append(L.argMax())    # choose label with highest likelyhood
    return guesses
    

    
      
