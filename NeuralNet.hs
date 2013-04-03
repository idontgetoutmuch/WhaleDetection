module NeuralNet where

import MatrixPlus

import Numeric.LinearAlgebra as N
import Test.QuickCheck

class NeuralNet n where
  evaluate
    :: n        -- ^ The neural net
    -> [Double] -- ^ The input pattern
    -> [Double] -- ^ The output pattern
  train
    :: n        -- ^ The neural net before training
    -> [Double] -- ^ The input pattern
    -> [Double] -- ^ The target pattern
    -> n        -- ^ The neural net after training
