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


Sadly we have to use an inefficient calculation to multiply matrices; see this [email][ADMatrixMult] for further details.

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
> propagate layerJ layerK = {- trace ("weights = " ++ show weights ++ "\n" ++
>                                  "1:layerJOut = " ++ show (1:layerJOut) ++ "\n" ++
>                                  "a = " ++ show a ++ "\n" ++
>                                  "map f a = " ++ show (map f a)) $ -}
>                           result
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
>     l = fromIntegral $ V.length expectedDigits
>     a = V.sum $ V.zipWith (\expectedDigit input -> costFn nDigits expectedDigit input net)
>                           expectedDigits inputs
>     b = (/2) $ sum $ map (^2) $
>         concat $ concat $
>         map stripBias $
>         extractWeights net
>     stripBias xss = map (drop 1) xss
>
>     cost = (a + delta * b) / l

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
> stepOnceTotal nDigits gamma y x net = {- trace ("net = " ++ show (extractWeights net) ++ "\n" ++
>                                      "del = " ++ show (extractWeights $ delTotalCostNN y x net)) $ -}
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

> w1, w2 :: [[Double]]
> w1  = randomWeightMatrix 2 2
> w2  = randomWeightMatrix 3 2

> initNet1 :: BackpropNet Double
> initNet1 = buildBackpropNet lRate [w1, w2] (ActivationFunction logit)
>
> labels :: V.Vector Int
> labels = V.map (round . fst) createSample

FIXME: Mixing vectors and lists seems a bit naff

> inputs :: V.Vector [Double]
> inputs = V.map (return . snd) createSample

> estimates :: (Floating a, Ord a, Show a) =>
>              V.Vector Int ->
>              V.Vector [a] ->
>              BackpropNet a ->
>              [BackpropNet a]
> estimates y x = gradientDescent $
>                 \theta -> totalCostNN 2 y (V.map (map auto) x) theta

> test1a :: IO ()
> test1a = do
>
>   let fs' = estimates labels inputs initNet1
>   mapM_ putStrLn $ map show $ map extractWeights $ take 10 $ drop 20 fs'
>   mapM_ putStrLn $ map show $ map (totalCostNN 2 labels inputs) $ take 10 $ drop 20 fs'
>   let phi' = extractWeights $ head $ drop 1000 fs'
>   putStrLn $ show phi'
>   putStrLn $ show $ evalNeuralNet (head $ drop 1000 fs') [0.1]
>   putStrLn $ show $ evalNeuralNet (head $ drop 1000 fs') [0.9]

Example III
-----------

> c, d :: Double
> c          = 15
> d          = 8
> sample2, sample3 :: [Double]
> sample2 = betas nSamples c d
> sample3 = betas nSamples d c

> mixSamples' :: Num t => [[a]] -> [(t, a)]
> mixSamples' xss = concat $ transpose $
>                   zipWith (\n xs -> map (n,) xs)
>                           (map fromIntegral [0..])
>                           xss
> sample02, sample03, sample12, sample13 :: [(Double, Double)]
> sample02 = [(x, y) | x <- sample0, y <- sample2]
> sample03 = [(x, y) | x <- sample0, y <- sample3]
> sample12 = [(x, y) | x <- sample1, y <- sample2]
> sample13 = [(x, y) | x <- sample1, y <- sample3]

> createSample' :: forall t. Num t => V.Vector (t, (Double, Double))
> createSample' = V.fromList $ take 512 $ mixSamples' [ sample02
>                                                     , sample03
>                                                     , sample12
>                                                     , sample13
>                                                     ]

> test2 :: IO ()
> test2 = do
>   let w1  = randomWeightMatrix 3 4
>       w2  = randomWeightMatrix 5 4
>       testNet1 = buildBackpropNet lRate [w1 {- , w2 -} ] (ActivationFunction logit)
>       testNet2 = buildBackpropNet lRate [w1, w2] (ActivationFunction logit)
>   let us = V.map fst createSample'
>   let vs = V.map ((\(x, y) -> [x, y]) . snd) createSample'
>   putStrLn "Time step 0 cost 1"
>   putStrLn $ show $ totalCostNN 4 (V.map round us) vs testNet1
>   putStrLn "Time step 0 cost 2"
>   putStrLn $ show $ totalCostNN 4 (V.map round us) vs testNet2
>   let f1s = iterate (stepOnceTotal 4 gamma us vs) testNet1
>       f2s = iterate (stepOnceTotal 4 gamma us vs) testNet2
>   putStrLn "Time step 1 cost 1"
>   putStrLn $ show $ totalCostNN 4 (V.map round us) vs (f1s!!1)
>   putStrLn "Time step 1 cost 2"
>   putStrLn $ show $ totalCostNN 4 (V.map round us) vs (f2s!!1)
>   putStrLn "Diff 1"
>   putStrLn $ show $ (totalCostNN 4 (V.map round us) vs (f1s!!1)) - (totalCostNN 4 (V.map round us) vs (f1s!!0))
>   putStrLn $ show $ (totalCostNN 4 (V.map round us) vs (f2s!!1)) - (totalCostNN 4 (V.map round us) vs (f2s!!0))
>   putStrLn "Cost 1s"
>   mapM_ putStrLn $ map show $ map (totalCostNN 4 (V.map round us) vs) $ take 10 $ drop 40 f1s
>   putStrLn "Cost 2s"
>   mapM_ putStrLn $ map show $ map (totalCostNN 4 (V.map round us) vs) $ take 10 $ drop 40 f2s
>   let g1s = drop 500 f1s
>       g2s = drop 500 f2s
>   mapM_ putStrLn $ map show $ map extractWeights $ take 2 g2s
>   putStrLn $ show $ evalNeuralNet (head g1s) [0.1, 0.9]
>   putStrLn $ show $ evalNeuralNet (head g1s) [0.9, 0.1]
>   putStrLn $ show $ evalNeuralNet (head g1s) [0.9, 0.9]
>   putStrLn $ show $ evalNeuralNet (head g2s) [0.1, 0.1]
>   putStrLn $ show $ evalNeuralNet (head g2s) [0.1, 0.9]
>   putStrLn $ show $ evalNeuralNet (head g2s) [0.9, 0.1]
>   putStrLn $ show $ evalNeuralNet (head g2s) [0.9, 0.9]

Appendix
--------

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


Backpropagation
---------------



The implementation below is a modified version of [MonadLayer].

We represent a layer as record consisting of the matrix of weights and
the activation function.

-- > data LayerOld =
-- >   LayerOld
-- >   {
-- >     layerWeightsOld  :: Matrix Double,
-- >     layerFunctionOld :: ActivationFunction
-- >   }

Our neural network consists of a list of layers together with a learning rate.

-- > data BackpropNetOld = BackpropNetOld
-- >     {
-- >       layersOld :: [LayerOld],
-- >       learningRateOld :: Double
-- >     }

The constructor function _buildBackPropnet_ does nothing more than
populate _BackPropNet_ checking that all the matrices of weights are
compatible.  It takes a learning rate, a list of matrices of weights
for each layer, a single common activation function and produce a
neural network.

-- > buildBackpropNetOld ::
-- >   Double ->
-- >   [Matrix Double] ->
-- >   ActivationFunction ->
-- >   BackpropNetOld
-- > buildBackpropNetOld learningRate ws f =
-- >   BackpropNetOld {
-- >       layersOld       = map buildLayer checkedWeights
-- >     , learningRateOld = learningRate
-- >     }
-- >   where checkedWeights = scanl1 checkDimensions ws
-- >         buildLayer w   = LayerOld { layerWeightsOld  = w
-- >                                , layerFunctionOld = f
-- >                                }
-- >         checkDimensions :: Matrix Double -> Matrix Double -> Matrix Double
-- >         checkDimensions w1 w2 =
-- >           if rows w1 == cols w2
-- >           then w2
-- >           else error "Inconsistent dimensions in weight matrix"

We keep a record of calculations at each layer in the neural network.

-- > data PropagatedLayerOld
-- >     = PropagatedLayerOld
-- >         {
-- >           propLayerInOld         :: ColumnVector Double,
-- >           propLayerOutOld        :: ColumnVector Double,
-- >           propLayerActFun'ValOld :: ColumnVector Double,
-- >           propLayerWeightsOld    :: Matrix Double,
-- >           propLayerActFunOld     :: ActivationFunction
-- >         }
-- >     | PropagatedSensorLayerOld
-- >         {
-- >           propLayerOutOld :: ColumnVector Double
-- >         }

We take a record of the calculations at one layer, a layer and produce
the record of the calculations at the next layer.

-- > propagateOld :: PropagatedLayerOld -> LayerOld -> PropagatedLayerOld
-- > propagateOld layerJ layerK = PropagatedLayerOld
-- >         {
-- >           propLayerInOld         = layerJOut,
-- >           propLayerOutOld        = mapMatrix f a,
-- >           propLayerActFun'ValOld = mapMatrix (diff f) a,
-- >           propLayerWeightsOld    = weights,
-- >           propLayerActFunOld     = layerFunctionOld layerK
-- >         }
-- >   where layerJOut = propLayerOutOld layerJ
-- >         weights   = layerWeightsOld layerK
-- >         a         = weights <> layerJOut
-- >         f :: Floating a => a -> a
-- >         f = activationFunction $ layerFunctionOld layerK

With this we can take an input to the neural network, the neural
network itself and produce a collection of records of the calculations
at each layer.

-- > propagateNetOld :: ColumnVector Double -> BackpropNetOld -> [PropagatedLayerOld]
-- > propagateNetOld input net = tail calcs
-- >   where calcs = scanl propagateOld layer0 (layersOld net)
-- >         layer0 = PropagatedSensorLayerOld $ validateInput net input
-- >
-- >         validateInput :: BackpropNetOld -> ColumnVector Double -> ColumnVector Double
-- >         validateInput net = validateInputValues . validateInputDimensions net
-- >
-- >         validateInputDimensions ::
-- >           BackpropNetOld ->
-- >           ColumnVector Double ->
-- >           ColumnVector Double
-- >         validateInputDimensions net input =
-- >           if got == expected
-- >           then input
-- >           else error ("Input pattern has " ++ show got ++ " bits, but " ++
-- >                       show expected ++ " were expected")
-- >           where got      = rows input
-- >                 expected = inputWidth $ head $ layersOld net
-- >
-- >         validateInputValues :: ColumnVector Double -> ColumnVector Double
-- >         validateInputValues input =
-- >           if (minimum ns >= 0) && (maximum ns <= 1)
-- >           then input
-- >           else error "Input bits outside of range [0,1]"
-- >           where
-- >             ns = toList ( flatten input )
-- >
-- >         inputWidth :: LayerOld -> Int
-- >         inputWidth = cols . layerWeightsOld


We keep a record of the back propagation calculations at each layer in
the neural network:

* The grad of the cost function with respect to the outputs of the layer.
* The grad of the cost function with respect to the weights of the layer.
* The value of the derivative of the activation function.
* The inputs to the layer.
* The outputs from the layer.
* The activation function.

-- > data BackpropagatedLayer = BackpropagatedLayer
-- >     {
-- >       backpropOutGrad    :: ColumnVector Double,
-- >       backpropWeightGrad :: Matrix Double,
-- >       backpropActFun'Val :: ColumnVector Double,
-- >       backpropIn         :: ColumnVector Double,
-- >       backpropOut        :: ColumnVector Double,
-- >       backpropWeights    :: Matrix Double,
-- >       backPropActFun     :: ActivationFunction
-- >     }

Propagate the inputs backward through this layer to produce an output.

-- > backpropagate :: PropagatedLayerOld ->
-- >                  BackpropagatedLayer ->
-- >                  BackpropagatedLayer
-- > backpropagate layerJ layerK = BackpropagatedLayer
-- >     {
-- >       backpropOutGrad    = dazzleJ,
-- >       backpropWeightGrad = errorGrad dazzleJ f'aJ bpIn,
-- >       backpropActFun'Val = f'aJ,
-- >       backpropIn         = bpIn,
-- >       backpropOut        = propLayerOutOld layerJ,
-- >       backpropWeights    = propLayerWeightsOld layerJ,
-- >       backPropActFun     = propLayerActFunOld layerJ
-- >     }
-- >     where dazzleJ = (trans $ backpropWeights layerK) <> (dazzleK * f'aK)
-- >           dazzleK = backpropOutGrad layerK
-- >           f'aK    = backpropActFun'Val layerK
-- >           f'aJ    = propLayerActFun'ValOld layerJ
-- >           bpIn    = propLayerInOld layerJ

-- > errorGrad :: ColumnVector Double ->
-- >              ColumnVector Double ->
-- >              ColumnVector Double ->
-- >              Matrix Double
-- > errorGrad dazzle f'a input = (dazzle * f'a) <> trans input

-- > backpropagateFinalLayer :: PropagatedLayerOld ->
-- >                            ColumnVector Double ->
-- >                            BackpropagatedLayer
-- > backpropagateFinalLayer l t = BackpropagatedLayer
-- >     {
-- >       backpropOutGrad    = dazzle,
-- >       backpropWeightGrad = errorGrad dazzle f'a (propLayerInOld l),
-- >       backpropActFun'Val = f'a,
-- >       backpropIn         = propLayerInOld l,
-- >       backpropOut        = propLayerOutOld l,
-- >       backpropWeights    = propLayerWeightsOld l,
-- >       backPropActFun     = propLayerActFunOld l
-- >     }
-- >     where dazzle =  propLayerOutOld l - t
-- >           f'a    = propLayerActFun'ValOld l

Move backward (from right to left) through the neural network
i.e. this is backpropagation itself.

-- > backpropagateNet :: ColumnVector Double ->
-- >                     [PropagatedLayerOld] ->
-- >                     [BackpropagatedLayer]
-- > backpropagateNet target layers = scanr backpropagate layerL hiddenLayers
-- >   where hiddenLayers = init layers
-- >         layerL = backpropagateFinalLayer (last layers) target

Now that we know all the derivatives with respect to the weights in
every layer, we can create a new layer by moving one step in the
direction of steepest descent.

-- > update :: Double ->
-- >           BackpropagatedLayer ->
-- >           LayerOld
-- > update rate layer = LayerOld
-- >         {
-- >           layerWeightsOld = wNew,
-- >           layerFunctionOld = backPropActFun layer
-- >         }
-- >     where wOld = backpropWeights layer
-- >           delW = rate `scale` backpropWeightGrad layer
-- >           wNew = wOld - delW

Now we can train our network by taking a list of inputs and outputs
using each pair to move a step in the direction of steepest descent.

-- > train :: BackpropNetOld ->
-- >          [Double] ->
-- >          [Double] ->
-- >          BackpropNetOld
-- > train net input target =
-- >   BackpropNetOld { layersOld       = newLayers
-- >               , learningRateOld = rate
-- >               }
-- >   where newLayers            = map (update $ learningRateOld net) backpropagatedLayers
-- >         rate                 = learningRateOld net
-- >         backpropagatedLayers = backpropagateNet (listToColumnVector target) propagatedLayers
-- >         propagatedLayers     = propagateNetOld x net
-- >         x                    = listToColumnVector (1:input)

-- > trainOnePattern :: ([Double], Int) -> BackpropNetOld -> BackpropNetOld
-- > trainOnePattern trainingData net = train net input target
-- >   where input = fst trainingData
-- >         digit = snd trainingData
-- >         target = targets !! digit
-- >

-- > trainWithAllPatterns :: BackpropNetOld ->
-- >                         [([Double], Int)]
-- >                         -> BackpropNetOld
-- > trainWithAllPatterns = foldl' (flip trainOnePattern)

Testing / Debugging
-------------------

-- > costFnFudge :: (Floating a, Ord a, Show a) => Int -> [a] -> BackpropNet a -> a
-- > costFnFudge expectedDigit input net = 0.5 * sum (map (^2) diffs) + b
-- >   where
-- >     b = (/2) $ sum $ map (^2) $ concat $ map concat $ extractWeights net
-- >     predicted = evalNeuralNet net input
-- >     diffs = zipWith (-) [fromIntegral expectedDigit] {- (targets!!expectedDigit) -} predicted

-- > delCostFnFudge :: (Ord a, Floating a, Show a) =>
-- >                          Int ->
-- >                          [a] ->
-- >                          BackpropNet a ->
-- >                          BackpropNet a
-- > delCostFnFudge y x = grad f
-- >   where
-- >     f theta = costFnFudge y (map auto x) theta

-- > stepOnceFudge :: Double ->
-- >             Int ->
-- >             [Double] ->
-- >             BackpropNet Double ->
-- >             BackpropNet Double
-- > stepOnceFudge gamma y x theta =
-- >   theta + fmap (* (negate gamma)) (delCostFnFudge y x theta)

-- > evaluateBPN :: BackpropNetOld -> [Double] -> [Double]
-- > evaluateBPN net input = columnVectorToList $ propLayerOutOld $ last calcs
-- >   where calcs = propagateNetOld x net
-- >         x = listToColumnVector (1:input)

-- > evalOnePattern :: BackpropNetOld -> ([Double], Int) -> Int
-- > evalOnePattern net trainingData =
-- >   isMatch result target
-- >   where input = fst trainingData
-- >         target = snd trainingData
-- >         rawResult = evaluateBPN net input
-- >         result = interpret rawResult

-- > evalAllPatterns :: BackpropNetOld -> [([Double], Int)] -> [Int]
-- > evalAllPatterns = map . evalOnePattern

-- > main2 = do
-- >   let w1  = randomWeightMatrix (nRows * nCols + 1) nNodes 7
-- >       w2  = randomWeightMatrix nNodes nDigits 42
-- >       initialNet  = buildBackpropNetOld  lRate [w1, w2] tanhAS
-- >       testNet = buildBackpropNet lRate [[[0.1, 0.1]]] (ActivationFunction logit)
-- >       testNet' = buildBackpropNet lRate [[[0.1, 0.1], [0.1, 0.1]]] (ActivationFunction logit)
-- >
-- >       u = round $ fst $ V.head createSample
-- >       v = snd $ V.head createSample
-- >       us = V.map round ws
-- >       vs = V.map snd createSample
-- >       ws = V.map fst createSample
-- >       xs = V.map (V.cons 1.0 . V.singleton) ws

-- >   let vals :: V.Vector (Double, V.Vector Double)
-- >       vals = V.map (\(y, x) -> (y, V.fromList [1.0, x])) $ createSample

-- >   printf "Gradient of cost %s\n" $ show $ extractWeights $ delCostFnFudge u [v] testNet
-- >   printf "Gradient of cost %s\n" $ show $ extractWeights $ delCostFn u [v] testNet'
-- >   printf "Step once %s\n" $ show $ extractWeights $ stepOnceFudge lRate u [v] testNet
-- >   printf "Step once %s\n" $ show $ extractWeights $ stepOnce lRate u [v] testNet'
-- >
-- >   let foo' = V.scanl' (\s (u, v) -> stepOnceFudge lRate u [v] s) testNet
-- >                       (V.zip (V.map fromIntegral us) vs)
-- >   printf "Step many %s\n" $ show $ V.map extractWeights $ V.drop 790 foo'
-- >   let foo'' = V.scanl' (\s (u, v) -> stepOnce lRate u [v] s) testNet'
-- >                        (V.zip (V.map fromIntegral us) vs)
-- >   printf "Step many %s\n" $ show $ V.map extractWeights $ V.drop 790 foo''

-- >   putStrLn $ show $ extractWeights testNet
-- >
-- >   trainingData <- fmap (take 8000) readTrainingData
-- >   let finalNet = trainWithAllPatterns initialNet trainingData
-- >
-- >   testData <- fmap (take 1000) readTestData
-- >   putStrLn $ "Testing with " ++ show (length testData) ++ " images"
-- >   let results = evalAllPatterns finalNet testData
-- >   let score = fromIntegral (sum results)
-- >   let count = fromIntegral (length testData)
-- >   let percentage = 100.0 * score / count
-- >   putStrLn $ "I got " ++ show percentage ++ "% correct"
