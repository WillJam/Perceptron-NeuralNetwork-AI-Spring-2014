"""Learning and prediction functions for artificial neural networks."""

import collections
import common
import math
import random
import sys

# Throughout this file, layer 0 of a neural network is the inputs, layer 1
# is the first hidden layer, etc.; the last layer is the outputs.

class NeuralNetwork:
  """An artificial neural network.

  Fields:
    weights: a list of lists of lists of numbers, where
       weights[a][b][c] is the weight into unit b of layer a+1 from unit c in
         layer a
    num_hidden_layers: an integer, the number of hidden layers in the network
  """

  def __init__(self, weights=None):
    self.weights = weights
    if weights:
      self.num_hidden_layers = len(weights) - 1

  def get_unit_values(self, features):
    """Calculate the activation of each unit in a neural network.

    Args:
      features: a vector of feature values

    Returns:
      units, a list of lists of numbers, where
        units[a][b] is the activation of unit b in layer a
    """

    units = list()    
    units.append(features)

    for x in range(0, len(self.weights)):
      layer = list()
      for y in range(0, len(self.weights[x])):
        v = common.dot(units[x], self.weights[x][y])
        layer.append(self.activation(v))
          
      units.append(layer)

    # COMPLETE THIS IMPLEMENTATION
    #print units
    return units


  def predict(self, features):

    """Calculate the prediction of a neural network on one example

    Args:
      features: a vector of feature values

    Returns:
      A list of numbers, the predictions for each output of the network
          for the given example.
    """
    # COMPLETE THIS IMPLEMENTATION
    return self.get_unit_values(features)[-1]


  def calculate_errors(self, unit_values, outputs):
    """Calculate the backpropagated errors for an input to a neural network.

    Args:
      unit_values: unit activations, a list of lists of numbers, where
        unit_values[a][b] is the activation of unit b in layer a
      outputs: a list of correct output values (numbers)

    Returns:
      A list of lists of numbers, the errors for each hidden or output unit.
          errors[a][b] is the error for unit b in layer a+1.
    """
    p = unit_values[-1]

    errors = []
    errout = []
    
    for x in range(0, len(p)):
      errout.append(p[x] * (1 - p[x]) * (outputs[x] - p[x]))

    errors.append(errout)
    
    for y in range(len(self.weights) -1, 0, -1):
      errhidden = []
      #print len(unit_values[y])
      for z in range(0, len(unit_values[y])):
        hid = unit_values[y][z]
        hidToOut = 0
        for j in range(0,len(self.weights[y])):
          hidToOut += self.weights[y][j][z] * errors[0][j]
           
        errhidden.append(hid *(1 - hid) * hidToOut)
      errors.insert(0, errhidden)
    
    #print errors 
    return errors

  def activation(self, v):
    return 1 / (1 + math.exp(-v))


  def init_weights(self, num_init_units, num_hidden_layers, num_hidden):
    weights = []
    #initilize
    weights.append([])
    
    for x in range(0, num_hidden):
      units = []
      for y in range(0, num_init_units):
        units.append(random.random())
      weights[0].append(units)
    #mid layers
    for x in range(1, num_hidden_layers):
      weights.append([])
      for z in range(0, num_hidden):
        units = []
        for y in range(0, num_hidden):
          units.append(random.random())
        weights[x].append(units)

    outputweights = []

    # out put layers
    for z in range(0, num_hidden):
      outputweights.append(random.random())

    weights.append([outputweights])

    self.weights = weights
  def learn(self,
      data,
      num_hidden=16,
      max_iterations=1000,
      learning_rate=1,
      num_hidden_layers=1):
    """Learn a neural network from data.

    Sets the weights for a neural network base training data.

    Args:
      data: a list of pairs of input and output vectors, both lists of numbers.
      num_hidden: the number of hidden units to use.
      max_iterations: the max number of iterations to train before stopping.
      learning_rate: a scaling factor to apply to each weight update.
      num_hidden_layers: the number of hidden layers to use.
        Unless you are doing the extra credit, you can ignore this parameter.

    Returns:
      This object, once learned.
    """
    # COMPLETE THIS IMPLEMENTATION
    # Use predict, get_unit_values, and calculate_errors
    # in your implementation!
    
    
    notconverged = True
    iters = 0

    self.init_weights(len(data[0][0]), num_hidden_layers, num_hidden)
    """
    for x in self.weights:
      for y in x:
        print y
    """

    while(notconverged):
      iters += 1
      for x in data:
        feats = x[0]
        y = x[1]
        
        unit_values = self.get_unit_values(feats)
        errors = self.calculate_errors(unit_values, y)
        
        for l in range(0, len(unit_values) - 1):
          for n in range(0, len(unit_values[l])):
            for x in range(0, len(unit_values[l + 1])):
              self.weights[l][x][n] += unit_values[l][n] * errors[l][x]
                  
      if all(z[1] == self.predict(z[0]) for z in data):
          break  
      if iters > max_iterations:
        break
    return self






  
