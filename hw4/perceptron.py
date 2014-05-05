"""Learning and prediction functions for perceptrons."""

import common
import copy

class NotConverged(Exception):
  """An exception raised when the perceptron training isn't converging."""


class Perceptron:
  def __init__(self, weights=None):
    self.weights = weights

  def learn(self, examples, max_iterations=100):
    """Learn a perceptron from [([feature], class)].

    Set the weights member variable to a list of numbers corresponding
    to the weights learned by the perceptron algorithm from the training
    examples.

    The number of weights should be one more than the number of features
    in each example.

    Args:
      examples: a list of pairs of a list of features and a class variable.
        Features should be numbers, class should be 0 or 1.
      max_iterations: number of iterations to train.  Gives up afterwards

    Raises:
      NotConverged, if training did not converge within the provided number
        of iterations.

    Returns:
      This object
    """
    
    # COMPLETE THIS IMPLEMENTATION
    self.weights = [0]
    for x in range(0, len(examples[0])):
      self.weights.append(0)

    notconverged = True
    
    iters = 0
    while(notconverged):
      iters += 1
      
      for z in examples:
        x = z[0]
        y = z[1]
        p = self.predict(x)
        
        x.append(1)
        
        self.weights = common.scale_and_add(self.weights, ((y - p)), x)
        
        del x[-1]
        
      if all(z[1] == self.predict(z[0]) for z in examples):
          break

      if iters > max_iterations:
        raise NotConverged()
      
    return self

  def predict(self, features):
    """Return the prediction given perceptron weights on an example.

    Args:
      features: A vector of features, [f1, f2, ... fn], all numbers

    Returns:
      1 if w1 * f1 + w2 * f2 + ... * wn * fn + t > 0
      0 otherwise
    """
    features.append(1)
    
    ret = 0
    value = common.dot(self.weights, features)
    
    if(value >0):
      ret = 1

    del features[-1]
    
    return ret

