% Neural Networks and Automated Differentiation
% Dominic Steinitz
% 4th April 2013

Introduction
------------

Neural networks are a method for classifying data based on a theory of
how biological systems operate. They can also be viewed as a
generalization of logistic regression. A method for determining the
coefficients of a given model, backpropagation, was developed in the
1970's and rediscovered in the 1980's.

The article "A Functional Approach to Neural Networks" in the [Monad
Reader][MonadReader] shows how to use a neural network to classify
handwritten digits in the [MNIST database][MNIST] using backpropagation.

  [MonadReader]: http://themonadreader.files.wordpress.com/2013/03/issue214.pdf
  [MNIST]: http://yann.lecun.com/exdb/mnist/
  [LeCunCortesMnist]: http://yann.lecun.com/exdb/mnist/

The reader is struck by how similar [backpropagation][Backpropagation]
is to [automatic differentiation][AutomaticDifferentiation]. The
reader may not therefore be surprised to find that this observation
had been made before: [Domke2009a][Domke2009a]. Indeed as Dan Piponi
observes: "the grandaddy machine-learning algorithm of them all,
back-propagation, is nothing but steepest descent with reverse mode
automatic differentiation".

  [Backpropagation]: http://en.wikipedia.org/wiki/Backpropagation
  [AutomaticDifferentiation]: http://en.wikipedia.org/wiki/Automatic_differentiation
  [Domke2009a]: http://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/

Neural Networks
---------------

We can view neural nets or at least a multi layer perceptron as a
generalisation of (multivariate) linear logistic regression.

We follow [@rojas1996neural;@Bishop:2006:PRM:1162264]. We are given a training set:

$$
\{(\boldsymbol{x}_0, \boldsymbol{y}_0), (\boldsymbol{x}_1, \boldsymbol{y}_1), \ldots, (\boldsymbol{x}_p, \boldsymbol{y}_p)\}
$$

of pairs of $n$-dimensional and $m$-dimensional vectors called the
input and output patterns in Machine Learning parlance. We wish to
build a neural network model using this training set.

A neural network model (or at least the specific model we discuss: the
[multi-layer perceptron][MultiLayerPerceptron]) consists of a sequence
of transformations. The first transformation creates weighted sums of
the inputs.

  [MultiLayerPerceptron]: http://en.wikipedia.org/wiki/Multilayer_perceptron

$$
a_j^{(1)} = \sum_{i=1}^{K_0} w^{(1)}_{ij}x_i + w_{0j}^{(1)}
$$

where $K_0 \equiv n$ is the size of the input vector and there are $j
= 1,\ldots,K_1$ neurons in the so called first hidden layer of the
network. The weights are unknown.

The second transformation then applies a non-linear activation
function $f$ to each $a_j$ to give the output from the $j$-th neuron in
the first hidden layer.

$$
z_j^{(1)} = f(a_j^{(1)})
$$

Typically, $f$ is chosen to be $\tanh$ or the logistic function. Note
that if it were chosen to be the identity then our neural network
would be the same as a multivariate linear logistic regression.

We now repeat these steps for the second hidden layer:

\begin{align*}
a_j^{(2)} &=  \sum_{i=1}^{K_1} w^{(2)}_{ij}z_i^{(1)} + w_{0j}^{(2)} \\
z_j^{(2)} &= f(a_j^{(2)})
\end{align*}

Ultimately after we applied $L-1$ transformations (through $L-1$ hidden
layers) we produce some output:

\begin{align*}
a_j^{(L)}       &= \sum_{i=1}^{K_{L-1}} w^{(L-1)}_{ij}x_i + w_{0j}^{(L-1)} \\
\hat{y}_j       &= f(a_j^{(L)})
\end{align*}

We show an example neural in the diagram below.

```{.dia width='500'}
import NnClassifierDia
dia = nn
```

The input layer has 7 nodes. There are 2 hidden layers, the first has
3 nodes and the second has 5. The output layer has 3 nodes.

We are also
given a cost function:

$$
E(\boldsymbol{w}; \boldsymbol{x}, \boldsymbol{y}) = \frac{1}{2}\|(\hat{\boldsymbol{y}} - \boldsymbol{y})\|^2
$$

where $\hat{\boldsymbol{y}}$ is the predicted output of the neural net
and $\boldsymbol{y}$ is the observed output.

As with logistic regression, our goal is to find weights for the
neural network which minimises this cost function. We initialise the
weights to some small non-zero amount and then use the method of
steepest descent (aka gradient descent). The idea is that if $f$ is a
function of several variables then to find its minimum value, one
ought to take a small step in the direction in which it is decreasing
most quickly and repeat until no step in any direction results in a
decrease. The analogy is that if one is walking in the mountains then
the quickest way down is to walk in the direction which goes down most
steeply. Of course one get stuck at a local minimum rather than the
global minimum but from a machine learning point of view this may be
acceptable; alternatively one may start at random points in the search
space and check they all give the same minimum.

We therefore need calculate the gradient of the loss function with
respect to the weights (since we need to minimise the cost
function). In other words we need to find:

$$
\nabla E(\boldsymbol{x}) \equiv (\frac{\partial E}{\partial w_1}, \ldots, \frac{\partial E}{\partial w_n})
$$

Once we have this we can take our random starting position and move
down the steepest gradient:

$$
w'_i = w_i - \gamma\frac{\partial E}{\partial w_i}
$$

where $\gamma$ is the step length known in machine learning parlance
as the learning rate.

Haskell Foreword
----------------

Some pragmas and imports required for the example code.

> {-# LANGUAGE RankNTypes                #-}
> {-# LANGUAGE DeriveFunctor             #-}
> {-# LANGUAGE DeriveFoldable            #-}
> {-# LANGUAGE DeriveTraversable         #-}
> {-# LANGUAGE ScopedTypeVariables       #-}
> {-# LANGUAGE TupleSections             #-}
> {-# LANGUAGE NoMonomorphismRestriction #-}

> {-# OPTIONS_GHC -Wall                     #-}
> {-# OPTIONS_GHC -fno-warn-name-shadowing  #-}
> {-# OPTIONS_GHC -fno-warn-type-defaults   #-}
> {-# OPTIONS_GHC -fno-warn-unused-do-bind  #-}
> {-# OPTIONS_GHC -fno-warn-missing-methods #-}

> module NeuralNet where

> import Numeric.AD
> import Numeric.AD.Types

> import Data.Traversable (Traversable)
> import Data.Foldable (Foldable)
> import Data.List
> import Data.List.Split
> import System.Random
> import qualified Data.Vector as V

> import Control.Monad
> import Control.Monad.State

> import Data.Random ()
> import Data.Random.Distribution.Beta
> import Data.Random.Distribution.Uniform
> import Data.RVar

> import Data.Word
> import Data.Bits
> import qualified Data.ByteString.Lazy as BL
> import Data.Binary.Get
> import Text.Printf

Logistic Regression Redux
-------------------------

Let us first implement logistic regression. This will give us a
reference against which to compare the equivalent solution expressed
as a neural network.

Instead of maximimizing the log likelihood, we will minimize a cost function.

> cost :: Floating a => V.Vector a -> a -> V.Vector a -> a
> cost theta y x = 0.5 * (y - yhat)^2
>   where
>     yhat = logit $ V.sum $ V.zipWith (*) theta x

> logit :: Floating a =>
>          a -> a
> logit x = 1 / (1 + exp (negate x))

We add a regularization term into the total cost so that the
parameters do not grow too large. Note that we do not regularize over
the bias.

> delta :: Floating a => a
> delta = 0.01

> totalCost :: Floating a =>
>              V.Vector a ->
>              V.Vector a ->
>              V.Vector (V.Vector a) ->
>              a
> totalCost theta y x = (a + delta * b) / l
>   where
>     l = fromIntegral $ V.length y
>     a = V.sum $ V.zipWith (cost theta) y x
>     b = (/2) $ V.sum $ V.map (^2) $ V.drop 1 theta

We determine the gradient of the regularized cost function.

> delTotalCost :: Floating a =>
>                 V.Vector a ->
>                 V.Vector (V.Vector a) ->
>                 V.Vector a ->
>                 V.Vector a
> delTotalCost y x = grad f
>   where
>     f theta = totalCost theta (V.map auto y) (V.map (V.map auto) x)

And finally we can apply [gradient descent][GradientDescent].

  [GradientDescent]: http://en.wikipedia.org/wiki/Gradient_descent

> gamma :: Double
> gamma = 0.4

> stepOnceCost :: Floating a =>
>                  a ->
>                  V.Vector a ->
>                  V.Vector (V.Vector a) ->
>                  V.Vector a ->
>                  V.Vector a
> stepOnceCost gamma y x theta =
>   V.zipWith (-) theta (V.map (* gamma) $ del theta)
>     where
>       del = delTotalCost y x

Neural Network Representation
-----------------------------

Let us borrow, generalize and prune the data structures used in
["A Functional Approach to Neural Networks"][MonadReader].
Some of the fields in the borrowed data structures are probably no
longer necessary given that we are going to use automated
differentiation rather than backpropagation. Caveat lector!

The activation function itself is a function which takes any type in
the _Floating_ class to the same type in the _Floating_ class e.g. _Double_.

> newtype ActivationFunction =
>   ActivationFunction
>   {
>     activationFunction :: Floating a => a -> a
>   }

A neural network is a collection of layers.

> data Layer a =
>   Layer
>   {
>     layerWeights  :: [[a]],
>     layerFunction :: ActivationFunction
>   } deriving (Functor, Foldable, Traversable)

> data BackpropNet a = BackpropNet
>     {
>       layers       :: [Layer a],
>       learningRate :: Double
>     } deriving (Functor, Foldable, Traversable)

We need some helper functions to build our neural network and to
extract information from it.

> buildBackpropNet ::
>   Double ->
>   [[[a]]] ->
>   ActivationFunction ->
>   BackpropNet a
> buildBackpropNet learningRate ws f =
>   BackpropNet {
>       layers       = map buildLayer checkedWeights
>     , learningRate = learningRate
>     }
>   where checkedWeights = scanl1 checkDimensions ws
>         buildLayer w   = Layer { layerWeights  = w
>                                 , layerFunction = f
>                                 }
>         checkDimensions :: [[a]] -> [[a]] -> [[a]]
>         checkDimensions w1 w2 =
>           if 1 + length w1 == length (head w2)
>           then w2
>           else error $ "Inconsistent dimensions in weight matrix\n" ++
>                         show (length w1)        ++ "\n" ++
>                         show (length w2)        ++ "\n" ++
>                         show (length $ head w1) ++ "\n" ++
>                         show (length $ head w2)

> extractWeights :: BackpropNet a -> [[[a]]]
> extractWeights x = map layerWeights $ layers x

In order to undertake gradient descent on the data structure in which
we store a neural network, _BackpropNet_, it will be convenient to be
able to add such structures together point-wise.

> instance Num a => Num (Layer a) where
>   (+) = addLayer
>
> addLayer :: Num a => Layer a -> Layer a -> Layer a
> addLayer x y = Layer { layerWeights  = zipWith (zipWith (+)) (layerWeights x) (layerWeights y)
>                       , layerFunction = layerFunction x
>                       }

> instance Num a => Num (BackpropNet a) where
>   (+) = addBPN

> addBPN :: Num a => BackpropNet a -> BackpropNet a -> BackpropNet a
> addBPN x y = BackpropNet { layers = zipWith (+) (layers x) (layers y)
>                           , learningRate = learningRate x
>                           }

We store information about updating of output values in each layer in
the neural network as we move forward through the network (aka forward
propagation).

> data PropagatedLayer a
>     = PropagatedLayer
>         {
>           propLayerIn         :: [a],
>           propLayerOut        :: [a],
>           propLayerWeights    :: [[a]],
>           propLayerActFun     :: ActivationFunction
>         }
>     | PropagatedSensorLayer
>         {
>           propLayerOut :: [a]
>         } deriving (Functor, Foldable, Traversable)


Sadly we have to use an inefficient calculation to multiply matrices;
see this [email][ADMatrixMult] for further details.

  [ADMatrixMult]: http://www.haskell.org/pipermail/haskell-cafe/2013-April/107543.html

> matMult :: Num a => [[a]] -> [a] -> [a]
> matMult m v = result
>   where
>     lrs = map length m
>     l   = length v
>     result = if all (== l) lrs
>              then map (\r -> sum $ zipWith (*) r v) m
>              else error $ "Matrix has rows of length " ++ show lrs ++
>                           " but vector is of length " ++ show l

Now we can propagate forwards. Note that the code from which this is
borrowed assumes that the inputs are images which are $m \times m$
pixels each encoded using a grayscale, hence the references to bits
and the check that values lie in the range $0 \leq x \leq 1$.

> propagateNet :: (Floating a, Ord a, Show a) => [a] -> BackpropNet a -> [PropagatedLayer a]
> propagateNet input net = tail calcs
>   where calcs = scanl propagate layer0 (layers net)
>         layer0 = PropagatedSensorLayer $ validateInput net input
>
>         validateInput net = validateInputValues . validateInputDimensions net
>
>         validateInputDimensions net input =
>           if got == expected
>           then input
>           else error ("Input pattern has " ++ show got ++ " bits, but " ++
>                       show expected ++ " were expected")
>           where got      = length input
>                 expected = (+(negate 1)) $
>                            length $
>                            head $
>                            layerWeights $
>                            head $
>                            layers net
>
>         validateInputValues input =
>           if (minimum input >= 0) && (maximum input <= 1)
>           then input
>           else error "Input bits outside of range [0,1]"

Note that we add a 1 to the inputs to each layer to give the bias.

> propagate :: (Floating a, Show a) => PropagatedLayer a -> Layer a -> PropagatedLayer a
> propagate layerJ layerK = result
>   where
>     result =
>       PropagatedLayer
>         {
>           propLayerIn         = layerJOut,
>           propLayerOut        = map f a,
>           propLayerWeights    = weights,
>           propLayerActFun     = layerFunction layerK
>         }
>     layerJOut = propLayerOut layerJ
>     weights   = layerWeights layerK
>     a = weights `matMult` (1:layerJOut)
>     f :: Floating a => a -> a
>     f = activationFunction $ layerFunction layerK

> evalNeuralNet :: (Floating a, Ord a, Show a) => BackpropNet a -> [a] -> [a]
> evalNeuralNet net input = propLayerOut $ last calcs
>   where calcs = propagateNet input net

We define a cost function.

> costFn :: (Floating a, Ord a, Show a) =>
>           Int ->
>           Int ->
>           [a] ->
>           BackpropNet a ->
>           a
> costFn nDigits expectedDigit input net = 0.5 * sum (map (^2) diffs)
>   where
>     predicted = evalNeuralNet net input
>     diffs = zipWith (-) ((targets nDigits)!!expectedDigit) predicted

> targets :: Floating a => Int -> [[a]]
> targets nDigits = map row [0 .. nDigits - 1]
>   where
>     row m = concat [x, 1.0 : y]
>       where
>         (x, y) = splitAt m (take (nDigits - 1) $ repeat 0.0)

And the gradient of the cost function. Note that both the cost
function and its gradient are parameterised over the inputs and the
output label.

> delCostFn :: (Ord a, Floating a, Show a) =>
>                          Int ->
>                          [a] ->
>                          BackpropNet a ->
>                          BackpropNet a
> delCostFn y x = grad f
>   where
>     f theta = costFn 2 y (map auto x) theta

Now we can implement (stochastic) gradient descent.

> stepOnce :: Double ->
>             Int ->
>             [Double] ->
>             BackpropNet Double ->
>             BackpropNet Double
> stepOnce gamma y x net =
>   net + fmap (* (negate gamma)) (delCostFn y x net)

If instead we would rather perform gradient descent over the whole
training set (rather than stochastically) then we can do so. Note that
we do not regularize the weights for the biases.

> totalCostNN :: (Floating a, Ord a, Show a) =>
>                Int ->
>                V.Vector Int ->
>                V.Vector [a] ->
>                BackpropNet a ->
>                a
> totalCostNN nDigits expectedDigits inputs net = cost
>   where
>     cost = (a + delta * b) / l
>
>     l = fromIntegral $ V.length expectedDigits
>
>     a = V.sum $ V.zipWith (\expectedDigit input -> costFn nDigits expectedDigit input net)
>                           expectedDigits inputs
>
>     b = (/(2 * m)) $ sum $ map (^2) ws
>
>     m = fromIntegral $ length ws
>
>     ws = concat $ concat $
>          map stripBias $
>          extractWeights net
>
>     stripBias xss = map (drop 1) xss

> delTotalCostNN :: (Floating a, Ord a, Show a) =>
>                   Int ->
>                   V.Vector Int ->
>                   V.Vector [a] ->
>                   BackpropNet a ->
>                   BackpropNet a
> delTotalCostNN nDigits expectedDigits inputs = grad f
>   where
>     f net = totalCostNN nDigits expectedDigits (V.map (map auto) inputs) net

> stepOnceTotal :: Int ->
>                  Double ->
>                  V.Vector Int ->
>                  V.Vector [Double] ->
>                  BackpropNet Double ->
>                  BackpropNet Double
> stepOnceTotal nDigits gamma y x net =
>   net + fmap (* (negate gamma)) (delTotalCostNN nDigits y x net)

Example I
---------

Let's try it out. First we need to generate some data.  Rather
arbitrarily let us create some populations from the `beta`
distribution.

> betas :: Int -> Double -> Double -> [Double]
> betas n a b =
>   fst $ runState (replicateM n (sampleRVar (beta a b))) (mkStdGen seed)
>     where
>       seed = 0

We can plot the populations we wish to distinguish by sampling.

> a, b :: Double
> a          = 15
> b          = 6
> nSamples :: Int
> nSamples   = 100000
>
> sample0, sample1 :: [Double]
> sample0 = betas nSamples a b
> sample1 = betas nSamples b a

> mixSamples :: [Double] -> [Double] -> [(Double, Double)]
> mixSamples xs ys = unfoldr g ((map (0,) xs), (map (1,) ys))
>   where
>     g ([], [])         = Nothing
>     g ([],  _)         = Nothing
>     g ( _, [])         = Nothing
>     g ((x:xs), (y:ys)) = Just $ (x, (y:ys, xs))

> createSample :: V.Vector (Double, Double)
> createSample = V.fromList $ take 100 $ mixSamples sample1 sample0

> lRate :: Double
> lRate = 0.01
> actualTheta :: V.Vector Double
> actualTheta = V.fromList [0.0, 1.0]
> initTheta :: V.Vector Double
> initTheta = V.replicate (V.length actualTheta) 0.1

> test1 :: IO ()
> test1 = do
>
>   let testNet = buildBackpropNet lRate [[[0.1, 0.1], [0.1, 0.1]]] (ActivationFunction logit)

>   let vals :: V.Vector (Double, V.Vector Double)
>       vals = V.map (\(y, x) -> (y, V.fromList [1.0, x])) $ createSample
>
>   let gs = iterate (stepOnceCost gamma (V.map fst vals) (V.map snd vals)) initTheta
>       theta = head $ drop 1000 gs
>   printf "Logistic regression: theta_0 = %5.3f, theta_1 = %5.3f\n"
>          (theta V.! 0) (theta V.! 1)
>
>   let us = V.map (round . fst) createSample
>   let vs = V.map snd createSample
>   let fs = iterate (stepOnceTotal 2 gamma us (V.map return vs)) testNet
>       phi = extractWeights $ head $ drop 1000 fs
>   printf "Neural network: theta_00 = %5.3f, theta_01 = %5.3f\n"
>          (((phi!!0)!!0)!!0) (((phi!!0)!!0)!!1)
>   printf "Neural network: theta_10 = %5.3f, theta_11 = %5.3f\n"
>          (((phi!!0)!!1)!!0) (((phi!!0)!!1)!!1)

    [ghci]
    test1

Example II
----------

Now let's try a neural net with 1 hidden layer using the data we prepared earlier.

> w1, w2 :: [[Double]]
> w1  = randomWeightMatrix 2 2
> w2  = randomWeightMatrix 3 2

> initNet2 :: BackpropNet Double
> initNet2 = buildBackpropNet lRate [w1, w2] (ActivationFunction logit)
>
> labels :: V.Vector Int
> labels = V.map (round . fst) createSample

> inputs :: V.Vector [Double]
> inputs = V.map (return . snd) createSample

Instead of hand-crafting gradient descent, let us use the library
function as it performs better and is easier to implement.

> estimates :: (Floating a, Ord a, Show a) =>
>              V.Vector Int ->
>              V.Vector [a] ->
>              BackpropNet a ->
>              [BackpropNet a]
> estimates y x = gradientDescent $
>                 \theta -> totalCostNN 2 y (V.map (map auto) x) theta

Now we can examine the weights of our fitted neural net and apply it
to some test data.

> test2 :: IO ()
> test2 = do
>
>   let fs = estimates labels inputs initNet2
>   putStrLn $ show $ extractWeights $ head $ drop 1000 fs
>   putStrLn $ show $ evalNeuralNet (head $ drop 1000 fs) [0.1]
>   putStrLn $ show $ evalNeuralNet (head $ drop 1000 fs) [0.9]

Example III
-----------

Let's try a more sophisticated example and create a population of 4
groups which we measure with 2 variables.

> c, d :: Double
> c          = 15
> d          = 8
> sample2, sample3 :: [Double]
> sample2 = betas nSamples c d
> sample3 = betas nSamples d c

> mixSamples3 :: Num t => [[a]] -> [(t, a)]
> mixSamples3 xss = concat $ transpose $
>                   zipWith (\n xs -> map (n,) xs)
>                           (map fromIntegral [0..])
>                           xss
> sample02, sample03, sample12, sample13 :: [(Double, Double)]
> sample02 = [(x, y) | x <- sample0, y <- sample2]
> sample03 = [(x, y) | x <- sample0, y <- sample3]
> sample12 = [(x, y) | x <- sample1, y <- sample2]
> sample13 = [(x, y) | x <- sample1, y <- sample3]

> createSample3 :: forall t. Num t => V.Vector (t, (Double, Double))
> createSample3 = V.fromList $ take 512 $ mixSamples3 [ sample02
>                                                     , sample03
>                                                     , sample12
>                                                     , sample13
>                                                     ]

Rather annoyingly picking random weights seemed to give a local but
not global minimum. This may be a feature of having more nodes in the
hidden layer than in the input layer. By fitting a neural net with no
hidden layers to the data and using the outputs as inputs to fit
another neural net with no hidden layers, we can get a starting point
from which we can converge to the global minimum.

> w31, w32 :: [[Double]]
> w31 = [[-1.795626449637491,1.0687662199549477,0.6780994566671094],
>        [-0.8953174631646047,1.536931540024011,-1.7631220370122578],
>        [-0.4762453998497917,-2.005243268058972,1.2945899127545906],
>        [0.43019763097582875,-1.5711869072989957,-1.187180183656747]]
> w32 = [[-0.65116209142284,0.4837310591797774,-0.17870333721054968,
>         -0.6692619856605464,-1.062292154441557],
>        [-0.7521274440366631,-1.2071835415415136e-2,1.0078929981538551,
>         -1.3144243587577473,-0.5102027925579049],
>        [-0.7545728756863981,-0.4830112128458844,-1.2901624541811962,
>         1.0487049495446408,9.746209726152217e-3],
>        [-0.8576212271328413,-0.9035219951783956,-0.4034500456652809,
>         0.10091187689838758,0.781835908789879]]
>
> testNet3 :: BackpropNet Double
> testNet3 = buildBackpropNet lRate [w31, w32] (ActivationFunction logit)

> labels3 :: V.Vector Int
> labels3 = V.map (round . fst) createSample3
> inputs3 :: V.Vector [Double]
> inputs3 = V.map ((\(x, y) -> [x, y]) . snd) createSample3

Now we use the library _gradientDescent_ function to generate neural
net which ever better fit the data.

> estimates3 :: (Floating a, Ord a, Show a) =>
>               V.Vector Int ->
>               V.Vector [a] ->
>               BackpropNet a ->
>               [BackpropNet a]
> estimates3 y x = gradientDescent $
>                  \theta -> totalCostNN 4 y (V.map (map auto) x) theta

Finally we can fit a neural net and check that it correctly classifies
some data.

> test3 :: IO ()
> test3 = do
>   let fs = drop 100 $ estimates3 labels3 inputs3 testNet3
>   putStrLn $ show $ extractWeights $ head fs
>   putStrLn $ show $ evalNeuralNet (head fs) [0.1, 0.1]
>   putStrLn $ show $ evalNeuralNet (head fs) [0.1, 0.9]
>   putStrLn $ show $ evalNeuralNet (head fs) [0.9, 0.1]
>   putStrLn $ show $ evalNeuralNet (head fs) [0.9, 0.9]

Hand-written Digit Recognition
------------------------------

Let's now try the archetypal example of handwritten digit recognition
using the [MNIST database][MNIST]

FIXME: We can probably get the number of rows and columns from the
data itself.

Our neural net configuration. We wish to classify images which are $28
\times 28$ pixels into 10 digits using a single (hidden) layer neural
net with 20 nodes.

We (or rather the authors of the [MonadReader article][MonadReader])
represent an image as a record; the pixels are represented using an
8-bit grayscale.

> data Image = Image {
>       iRows    :: Int
>     , iColumns :: Int
>     , iPixels  :: [Word8]
>     } deriving (Eq, Show)

First we need some utilities to decode the data.

> deserialiseHeader :: Get (Word32, Word32, Word32, Word32, [[Word8]])
> deserialiseHeader = do
>   magicNumber <- getWord32be
>   imageCount <- getWord32be
>   r <- getWord32be
>   c <- getWord32be
>   packedData <- getRemainingLazyByteString
>   let len = fromIntegral (r * c)
>   let unpackedData = chunksOf len (BL.unpack packedData)
>   return (magicNumber, imageCount, r, c, unpackedData)

> readImages :: FilePath -> IO (Int, Int, [Image])
> readImages filename = do
>   content <- BL.readFile filename
>   let (_, _, r, c, unpackedData) = runGet deserialiseHeader content
>   return (fromIntegral r, fromIntegral c,
>           (map (Image (fromIntegral r) (fromIntegral c)) unpackedData))

> deserialiseLabels :: Get (Word32, Word32, [Word8])
> deserialiseLabels = do
>   magicNumber <- getWord32be
>   count <- getWord32be
>   labelData <- getRemainingLazyByteString
>   let labels = BL.unpack labelData
>   return (magicNumber, count, labels)

> readLabels :: FilePath -> IO [Int]
> readLabels filename = do
>   content <- BL.readFile filename
>   let (_, _, labels) = runGet deserialiseLabels content
>   return (map fromIntegral labels)

> uniforms :: Int -> [Double]
> uniforms n =
>   fst $ runState (replicateM n (sampleRVar stdUniform)) (mkStdGen seed)
>     where
>       seed = 0

We seed the weights in the neural with small random values; if we set
all the weights to 0 then the gradient descent algorithm might get stuck.

> randomWeightMatrix :: Int -> Int -> [[Double]]
> randomWeightMatrix numInputs numOutputs = y
>   where
>     y = chunksOf numInputs weights
>     weights = map (/ 100.0) $ uniforms (numOutputs * numInputs)

> nDigits, nNodes :: Int
> nDigits = 10
> nNodes  = 20

> main :: IO ()
> main = do
>   (nRows, nCols, trainingImages) <- readImages "train-images-idx3-ubyte"
>   trainingLabels                 <- readLabels "train-labels-idx1-ubyte"
>   let w1  = randomWeightMatrix (nRows * nCols + 1) nNodes
>       w2  = randomWeightMatrix (nNodes + 1) nDigits
>       testNet = buildBackpropNet lRate [w1, w2] (ActivationFunction logit)
>   let us :: V.Vector Int
>       us = V.take 10 $ V.fromList trainingLabels
>       exponent = bitSize $ head $ iPixels $ head trainingImages
>       normalizer = fromIntegral 2^exponent
>   let vs :: V.Vector [Double]
>       vs = V.take 10 $ V.fromList $
>            map (map ((/ normalizer) . fromIntegral) . iPixels) trainingImages
>   let fs = iterate (stepOnceTotal nDigits gamma us vs) testNet
>       phi = extractWeights $ head $ drop 1 fs
>   putStrLn $ show $ length (phi!!1)

-- >   printf "Neural network: theta_00 = %5.3f, theta_01 = %5.3f\n"
-- >          (((phi!!0)!!0)!!0) (((phi!!0)!!0)!!1)
-- >   printf "Neural network: theta_10 = %5.3f, theta_11 = %5.3f\n"
-- >          (((phi!!0)!!1)!!0) (((phi!!0)!!1)!!1)



We initialise our algorithm with arbitrary values.



-- > main :: IO ()
-- > main = do
-- >   let w1  = randomWeightMatrix (nRows * nCols + 1) nNodes 7
-- >       w2  = randomWeightMatrix nNodes nDigits 42
-- >       w1' :: (Random a, Floating a) => [[a]]
-- >       w1' = randomWeightMatrix' (nRows * nCols + 1) nNodes 7
-- >       w2' :: (Random a, Floating a) => [[a]]
-- >       w2' = randomWeightMatrix' nNodes nDigits 42
-- >       initialNet  = buildBackpropNetOld  lRate [w1, w2] tanhAS
-- >       testNet = buildBackpropNet lRate [[[0.1, 0.1]]] (ActivationFunction logit)
-- >       testNet' = buildBackpropNet lRate [[[0.1, 0.1], [0.1, 0.1]]] (ActivationFunction logit)
-- >
-- >   trainingData <- fmap (take 8000) readTrainingData
-- >
-- >   trainingLabels <- readLabels "train-labels-idx1-ubyte"
-- >   trainingImages <- readImages "train-images-idx3-ubyte"
-- >
-- >   let trainingData' :: RealFrac a => [LabelledImage a]
-- >       trainingData' = zip (map normalisedData' trainingImages) trainingLabels
-- >
-- >       u = round $ fst $ V.head createSample
-- >       v = snd $ V.head createSample
-- >       us = V.map round ws
-- >       vs = V.map snd createSample
-- >       ws = V.map fst createSample
-- >       xs = V.map (V.cons 1.0 . V.singleton) ws

-- >   let vals :: V.Vector (Double, V.Vector Double)
-- >       vals = V.map (\(y, x) -> (y, V.fromList [1.0, x])) $ createSample
-- >
-- >   let gs = iterate (stepOnceCost gamma (V.map fst vals) (V.map snd vals)) initTheta
-- >   printf "Working grad desc: %s\n" $ show $ take 10 $ drop 1000 gs
-- >
-- >   let fs = iterate (stepOnceTotal gamma us (V.map return vs)) testNet'
-- >   printf "Working grad desc: %s\n" $ show $ map extractWeights $ take 10 $ drop 1000 fs
-- >
-- >   error "Finished"




A labelled image contains the image and what this image actually
represents e.g. the image of the numeral 9 could and should be
represented by the value 9.

-- > type LabelledImage a = ([a], Int)
-- >
-- > -- | Inputs, outputs and targets are represented as column vectors instead of lists
-- > type ColumnVector a = Matrix a
-- >
-- >  -- | Convert a column vector to a list
-- > columnVectorToList :: (Ord a, Field a)
-- >     -- | The column vector to convert
-- >     => ColumnVector a
-- >     -- | The resulting list
-- >     -> [a]
-- > columnVectorToList = toList . flatten
-- >
-- > -- | Convert a list to a column vector
-- > listToColumnVector :: (Ord a, Field a)
-- >     -- | the list to convert
-- >     => [a]
-- >     -- | the resulting column vector
-- >     -> ColumnVector a
-- > listToColumnVector x = (len >< 1) x
-- >     where len = length x

-- > tanhAS :: ActivationFunction
-- > tanhAS = ActivationFunction
-- >     {
-- >       activationFunction = tanh
-- >     }

FIXME: This looks a bit yuk

-- > isMatch :: (Eq a) => a -> a -> Int
-- > isMatch x y =
-- >   if x == y
-- >   then 1
-- >   else 0

-- > interpret :: [Double] -> Int
-- > interpret v = fromJust (elemIndex (maximum v) v)

-- > readTrainingData ::  Floating a => IO [LabelledImage a]
-- > readTrainingData = do
-- >   trainingLabels <- readLabels "train-labels-idx1-ubyte"
-- >   trainingImages <- readImages "train-images-idx3-ubyte"
-- >   return $ zip (map normalisedData trainingImages) trainingLabels

-- > readTrainingData' ::  RealFrac a => IO [LabelledImage a]
-- > readTrainingData' = do
-- >   trainingLabels <- readLabels "train-labels-idx1-ubyte"
-- >   trainingImages <- readImages "train-images-idx3-ubyte"
-- >   return $ zip (map normalisedData' trainingImages) trainingLabels

-- > readTestData :: Floating a => IO [LabelledImage a]
-- > readTestData = do
-- >   putStrLn "Reading test labels..."
-- >   testLabels <- readLabels "t10k-labels-idx1-ubyte"
-- >   testImages <- readImages "t10k-images-idx3-ubyte"
-- >   return (zip (map normalisedData testImages) testLabels)

-- > normalisedData :: Floating a => Image -> [a]
-- > normalisedData image = map normalisePixel (iPixels image)
-- >   where
-- >     normalisePixel :: Floating a => Word8 -> a
-- >     normalisePixel p = (fromIntegral p) / 255.0

-- > normalisedData' :: RealFrac a => Image -> [a]
-- > normalisedData' image = map normalisePixel (iPixels image)
-- >   where
-- >     normalisePixel p = (fromIntegral p) / 255.0


