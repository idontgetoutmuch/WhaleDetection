{-# OPTIONS_GHC -Wall                    #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing #-}
{-# OPTIONS_GHC -fno-warn-type-defaults  #-}

module Backprop (
    BackpropNet
  , buildBackpropNet
  , logisticSigmoidAS
  , tanhAS
  , identityAS
  ) where

import MatrixPlus as P
import NeuralNet

import Numeric.LinearAlgebra as N
import Test.QuickCheck

-- | An individual layer in a neural network, prior to propagation
data Layer = Layer
    {
      -- The weights for this layer
      lW :: Matrix Double,
      -- The activation specification for this layer
      lAS :: ActivationSpec
    }

instance Show Layer where
    show layer = "w=" ++ show (lW layer) ++ ", activation spec=" ++ show (lAS layer)

inputWidth :: Layer -> Int
inputWidth = cols . lW

-- | An individual layer in a neural network, after propagation but prior to backpropagation
data PropagatedLayer
    = PropagatedLayer
        {
          -- The input to this layer
          pIn :: ColumnVector Double,
          -- The output from this layer
          pOut :: ColumnVector Double,
          -- The value of the first derivative of the activation function for this layer
          pF'a :: ColumnVector Double,
          -- The weights for this layer
          pW :: Matrix Double,
          -- The activation specification for this layer
          pAS :: ActivationSpec
        }
    | PropagatedSensorLayer
        {
          -- The output from this layer
          pOut :: ColumnVector Double
        }

instance Show PropagatedLayer where
    show (PropagatedLayer x y f'a w s) =
        "in=" ++ show x
        ++ ", out=" ++ show y
        ++ ", f'(a)=" ++ show f'a
        ++ ", w=" ++ show w
        ++ ", " ++ show s
    show (PropagatedSensorLayer x) = "out=" ++ show x

-- | Propagate the inputs through this layer to produce an output.
propagate :: PropagatedLayer -> Layer -> PropagatedLayer
propagate layerJ layerK = PropagatedLayer
        {
          pIn = x,
          pOut = y,
          pF'a = f'a,
          pW = w,
          pAS = lAS layerK
        }
  where x = pOut layerJ
        w = lW layerK
        a = w <> x
        f = asF ( lAS layerK )
        y = P.mapMatrix f a
        f' = asF' ( lAS layerK )
        f'a = P.mapMatrix f' a

-- | An individual layer in a neural network, after backpropagation
data BackpropagatedLayer = BackpropagatedLayer
    {
      -- Del-sub-z-sub-l of E
      bpDazzle :: ColumnVector Double,
      -- The error due to this layer
      bpErrGrad :: ColumnVector Double,
      -- The value of the first derivative of the activation
      --   function for this layer
      bpF'a :: ColumnVector Double,
      -- The input to this layer
      bpIn :: ColumnVector Double,
      -- The output from this layer
      bpOut :: ColumnVector Double,
      -- The weights for this layer
      bpW :: Matrix Double,
      -- The activation specification for this layer
      bpAS :: ActivationSpec
    }

instance Show BackpropagatedLayer where
    show layer =
        "dazzle=" ++ show (bpDazzle layer)
        ++ ", grad=" ++ show (bpErrGrad layer)
        ++ ", in=" ++ show (bpIn layer)
        ++ ", out=" ++ show (bpOut layer)
        ++ ", w=" ++ show (bpW layer)
        ++ ", activationFunction=?, activationFunction'=?"

backpropagateFinalLayer ::
    PropagatedLayer -> ColumnVector Double -> BackpropagatedLayer
backpropagateFinalLayer l t = BackpropagatedLayer
    {
      bpDazzle = dazzle,
      bpErrGrad = errorGrad dazzle f'a (pIn l),
      bpF'a = pF'a l,
      bpIn = pIn l,
      bpOut = pOut l,
      bpW = pW l,
      bpAS = pAS l
    }
    where dazzle =  pOut l - t
          f'a = pF'a l

errorGrad :: ColumnVector Double -> ColumnVector Double -> ColumnVector Double
    -> ColumnVector Double
errorGrad dazzle f'a input = (dazzle * f'a) <> trans input

-- | Propagate the inputs backward through this layer to produce an output.
backpropagate :: PropagatedLayer -> BackpropagatedLayer -> BackpropagatedLayer
backpropagate layerJ layerK = BackpropagatedLayer
    {
      bpDazzle = dazzleJ,
      bpErrGrad = errorGrad dazzleJ f'aJ (pIn layerJ),
      bpF'a = pF'a layerJ,
      bpIn = pIn layerJ,
      bpOut = pOut layerJ,
      bpW = pW layerJ,
      bpAS = pAS layerJ
    }
    where dazzleJ = wKT <> (dazzleK * f'aK)
          dazzleK = bpDazzle layerK
          wKT = trans ( bpW layerK )
          f'aK = bpF'a layerK
          f'aJ = pF'a layerJ

-- | Adjusting weights after backpropagation
update :: Double -> BackpropagatedLayer -> Layer
update rate layer = Layer
        {
          lW = wNew,
          lAS = bpAS layer
        }
    where wOld = bpW layer
          delW = rate `scale` bpErrGrad layer
          wNew = wOld - delW

-- | Building a network
data BackpropNet = BackpropNet
    {
      layers :: [Layer],
      learningRate :: Double
    } deriving Show

buildBackpropNet ::
  -- The learning rate
  Double ->
  -- The weights for each layer
  [Matrix Double] ->
  -- The activation specification (used for all layers)
  ActivationSpec ->
  -- The network
  BackpropNet
buildBackpropNet lr ws s = BackpropNet { layers=ls, learningRate=lr }
  where checkedWeights = scanl1 checkDimensions ws
        ls = map buildLayer checkedWeights
        buildLayer w = Layer { lW=w, lAS=s }

checkDimensions :: Matrix Double -> Matrix Double -> Matrix Double
checkDimensions w1 w2 =
  if rows w1 == cols w2
       then w2
       else error "Inconsistent dimensions in weight matrix"

propagateNet :: ColumnVector Double -> BackpropNet -> [PropagatedLayer]
propagateNet input net = tail calcs
  where calcs = scanl propagate layer0 (layers net)
        layer0 = PropagatedSensorLayer{ pOut=validatedInputs }
        validatedInputs = validateInput net input

validateInput :: BackpropNet -> ColumnVector Double -> ColumnVector Double
validateInput net = validateInputValues . validateInputDimensions net

validateInputDimensions ::
    BackpropNet ->
    ColumnVector Double ->
    ColumnVector Double
validateInputDimensions net input =
  if got == expected
       then input
       else error ("Input pattern has " ++ show got ++ " bits, but " ++ show expected ++ " were expected")
           where got = rows input
                 expected = inputWidth (head (layers net))

validateInputValues :: ColumnVector Double -> ColumnVector Double
validateInputValues input =
  if (min >= 0) && (max <= 1)
       then input
       else error "Input bits outside of range [0,1]"
       where min = minimum ns
             max = maximum ns
             ns = toList ( flatten input )

backpropagateNet ::
  ColumnVector Double -> [PropagatedLayer] -> [BackpropagatedLayer]
backpropagateNet target layers = scanr backpropagate layerL hiddenLayers
  where hiddenLayers = init layers
        layerL = backpropagateFinalLayer (last layers) target

-- | Define BackpropNet to be an instance of Neural Net
instance NeuralNet BackpropNet where
  evaluate = evaluateBPN
  train = trainBPN

evaluateBPN :: BackpropNet -> [Double] -> [Double]
evaluateBPN net input = columnVectorToList( pOut ( last calcs ))
  where calcs = propagateNet x net
        x = listToColumnVector (1:input)

trainBPN :: BackpropNet -> [Double] -> [Double] -> BackpropNet
trainBPN net input target = BackpropNet { layers=newLayers, learningRate=rate }
  where newLayers = map (update rate) backpropagatedLayers
        rate = learningRate net
        backpropagatedLayers = backpropagateNet (listToColumnVector target) propagatedLayers
        propagatedLayers = propagateNet x net
        x = listToColumnVector (1:input)

-- | A layer with suitable input and target vectors, suitable for testing.
data LayerTestData = LTD (ColumnVector Double) Layer (ColumnVector Double)
  deriving Show

-- | A layer with suitable input and target vectors, suitable for testing.
data TwoLayerTestData =
  TLTD (ColumnVector Double) Layer Layer (ColumnVector Double)
    deriving Show

-- | Common activation functions
data ActivationSpec = ActivationSpec
    {
      asF :: Double -> Double,
      asF' :: Double -> Double,
      desc :: String
    }

instance Show ActivationSpec where
    show = desc

identityAS :: ActivationSpec
identityAS = ActivationSpec
    {
      asF = id,
      asF' = const 1,
      desc = "identity"
    }

logisticSigmoidAS :: Double -> ActivationSpec
logisticSigmoidAS c = ActivationSpec
    {
        asF = logisticSigmoid c,
        asF' = logisticSigmoid' c,
        desc = "logistic sigmoid, c=" ++ show c
    }

instance Arbitrary ActivationSpec where
    arbitrary = return identityAS

logisticSigmoid :: (Field a, Floating a) => a -> a -> a
logisticSigmoid c a = 1 / (1 + exp((-c) * a))

logisticSigmoid' :: (Field a, Floating a) => a -> a -> a
logisticSigmoid' c a = (c * f a) * (1 - f a)
  where f = logisticSigmoid c


tanhAS :: ActivationSpec
tanhAS = ActivationSpec
    {
      asF = tanh,
      asF' = tanh',
      desc = "tanh"
    }

tanh' :: Double -> Double
tanh' x = 1 - (tanh x)^2

