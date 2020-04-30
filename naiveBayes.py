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
    labelTotal = len(trainingLabels)
    labelCounts = util.Counter()
    for y in trainingLabels:
      labelCounts[y] += 1
    priorDistrib = util.Counter()
    for y in self.legalLabels:
      priorDistrib[y] = labelCounts[y]/labelTotal
    
    self.priorDistrib = priorDistrib

    # Get count of feature = value for each label (used for computing conditional probabilities)
    # grid[label][feature][value] = count
    print(len(trainingData))
    featureValueGrid = util.Counter()
    for y in self.legalLabels:
      featureValueGrid[y] = util.Counter()
      for f in self.features:
        featureValueGrid[y][f] = util.Counter()
    for datum_index in range(0, len(trainingData)):
      datum = trainingData[datum_index]
      label = trainingLabels[datum_index]
      for feature in datum:
        value = datum[feature]
        featureValueGrid[label][feature][value] += 1

    # Get conditional probabilities
    # p(φ_i(x) = v | y = true) and p(φ_i(x) = v | y = false)
    probDistribTrue = util.Counter()    # list of p(φ_i(x) = v | y = true) for each value v of each feature i; 
    probDistribFalse = util.Counter()   # list of p(φ_i(x) = v | y = false) for each value v of each feature i
    for label in self.legalLabels:
      probDistribTrue[label] = util.Counter()    # list of features 
      probDistribFalse[label] = util.Counter()    # list of features 
      for feature in self.features:
        probDistribTrue[label][feature] = util.Counter()   # list of values
        probDistribFalse[label][feature] = util.Counter()   # list of values
        for value in featureValueGrid[label][feature]:
          if (featureValueGrid[label][feature][value] == 0):
            probDistribTrue[label][feature][value] = 0
          else:
            probDistribTrue[label][feature][value] = (featureValueGrid[label][feature][value])/(labelCounts[label])
          
          falseCount = 0
          for label2 in self.legalLabels:
            if (label2 == label):
              continue
            falseCount += featureValueGrid[label2][feature][value]
          if (falseCount == 0): 
            probDistribFalse[label][feature][value] = 0
          else:
            probDistribFalse[label][feature][value] = (falseCount)/(labelTotal - labelCounts[label])
    
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
          A1 = max(0.001, self.probDistribTrue[label][feature][value])
          A2 = max(0.001, self.priorDistrib[label])
          B1 = max(0.001, self.probDistribFalse[label][feature][value])
          B2 = max(0.001, (1.0 - self.priorDistrib[label]))
          # print (str(A1) + " * " + str (A2) + " / " + str(B1) + " * " + str(B2))
          L[label] *= ((A1 * A2) / (B1 * B2))
      guesses.append(L.argMax())    # choose label with highest likelyhood
    return guesses
    

    
      
