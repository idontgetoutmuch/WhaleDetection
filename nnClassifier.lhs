% Neural Networks and Automated Differentiation
% Dominic Steinitz
% 4th April 2013

Introduction
------------

Neural networks are a method for classifying data based on a theory of
how biological systems operate. They can also be viewed as a
generalization of logistic regression. A method for determining the
coefficients of a given model, backpropagation, was developed in the
1970's and rediscovered in the 1980'.

The article "A Functional Approach to Neural Networks" in the [Monad
Reader][MonadReader] shows how to use a neural network to classify
handwritten digits in the [MNIST database][MNIST] using backpropagation.

  [MonadReader]: http://themonadreader.files.wordpress.com/2013/03/issue21.pdf
  [MNIST]: http://yann.lecun.com/exdb/mnist/
  [LeCunCortesMnist]: http://yann.lecun.com/exdb/mnist/

The reader is struck by how similar [backpropagation][Backpropagation]
is to [automatic differentiation][AutomaticDifferentiation]. The
reader may not therefore be surprised to find that this observation
had been made before: [Domke2009a][Domke2009a]. Indeed as Dan Piponi
observes: "the grandaddy machine-learning algorithm of them all,
back-propagation, is nothing but steepest descent with reverse mode
automatic differentiation".

Automated differentiation was developed in the 1960's but even now
does not seem to be that widely used. Even experienced and
knowledgeable practitioners often assume it is either a finite
difference method or symbolic computation when it is neither.

  [Backpropagation]: http://en.wikipedia.org/wiki/Backpropagation
  [AutomaticDifferentiation]: http://en.wikipedia.org/wiki/Automatic_differentiation
  [Domke2009a]: http://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/

This article is divided into four parts:

* The first recalls logistic regression which can be viewed as a very
simple neural network (a single layer perceptron);

* The second explains the multi-layer perceptron neural network;

* The third summarises how backpropagation works;

* The last shows how backpropagation can be replaced by automated
differentation. Both techniques are applied to what appears to be the
standard [benchmark][LeCunCortesMnist] (MNIST).

Acknowledgements
---------------

The authors of the [MonadReader][MonadReader]: Amy de Buitléir,
Michael Russell and Mark Daly.

Haskell Foreword
----------------

Some pragmas and imports required for the example code.

> {-# LANGUAGE RankNTypes #-}
> {-# LANGUAGE DeriveFunctor #-}
> {-# LANGUAGE DeriveFoldable #-}
> {-# LANGUAGE DeriveTraversable #-}
> {-# LANGUAGE ScopedTypeVariables #-}
>
> {-# OPTIONS_GHC -Wall                    #-}
> {-# OPTIONS_GHC -fno-warn-name-shadowing #-}
> {-# OPTIONS_GHC -fno-warn-type-defaults  #-}

> module Main (main) where

> import Numeric.LinearAlgebra
> import Numeric.AD
> import Numeric.AD.Types
> import Data.Traversable (Traversable)
> import Data.Foldable (Foldable)
> import Data.List
> import Data.List.Split
> import System.Random

For use in the appendix.

> import Data.Word
> import qualified Data.ByteString.Lazy as BL
> import Data.Binary.Get
>
> import Data.Maybe
>
> import Numeric.GSL.Fitting.Linear

> import Data.Csv hiding (Field)
> import qualified Data.ByteString.Lazy as BS
> import System.IO
> import Data.Char
> import qualified Data.Vector as V

Multivariate Linear Logistic Regression
---------------------------------------

$$
y = tanh (\sum_{i=1}^{22} w_i * x_i + c)
$$

Let us consider linear regression first. We have

$$
\vec{y} = X\vec{w}
$$

We wish to minimise the loss function:

$$
\mathbb{L} = \vec{r}\cdot\vec{r}
$$

where

$$
\vec{r} = \vec{y} - X\vec{w}
$$

Differentiating:

$$
\frac{\partial\mathbb{L}}{\partial w_j} = 2 \sum_{i=1}^n r_i \frac{\partial r_i}{\partial w_j}
$$

We also have that:

$$
\frac{\partial r_i}{\partial w_j} = -X_{ij}
$$

Substituting:

$$
\frac{\partial\mathbb{L}}{\partial w_j} = 2 \sum_{i=1}^n (y_i - \sum_{k=1}^m X_{ik}w_k)(-X_{ij})
$$

The minimum of the loss function is reached when

$$
\frac{\partial \mathbb{L}}{\partial w_j} = 0
$$

Substituting again we have find that the values of $\vec{w} = \vec{\hat{w}}$ which
minimise the loss function satisfy

$$
2 \sum_{i=1}^n (y_i - \sum_{k=1}^m X_{ik}\hat{w}_k)(-X_{ij}) = 0
$$


  [LogisticRegression]: http://en.wikipedia.org/wiki/Logistic_regression

FIXME: We should reference the GLM book.

> myOptions :: DecodeOptions
> myOptions = defaultDecodeOptions {
>   decDelimiter = fromIntegral (ord ',')
>   }

> linReg :: IO ()
> linReg = do
>   vals <- {- fmap (V.take 12) $ -} withFile "/Users/dom/Downloadable/DataScienceLondon/Train.csv" ReadMode
>           (\h -> do c <- BS.hGetContents h
>                     let mvv :: Either String (V.Vector (V.Vector Double))
>                         mvv = decodeWith myOptions True c
>                     case mvv of
>                       Left s -> do putStrLn s
>                                    return V.empty
>                       Right vv -> return vv
>           )
>
>   let labels = fromList $ V.toList $ V.map (V.! 0) vals
>       inds  = V.map (V.drop 1) vals
>       rowsV = V.map (V.toList) xTrain
>       nRows = V.length rowsV
>       nCols = V.length $ V.head xTrain
>       featuress = V.map (V.splitAt 11) inds
>       tF x = log $ x + 1
>       xTrain = V.map (\fs -> V.zipWith (-) (V.map tF $ fst fs) (V.map tF $ snd fs))
>                      featuress
>       indVrs = (><) nRows nCols $ concat $ V.toList rowsV
>       (coeffs, covMat, _) = multifit indVrs labels
>       ests =  V.map (\x -> fst $ multifit_est (fromList x) coeffs covMat) rowsV
>   -- putStrLn $ show ests
>   -- putStrLn $ show labels
>   let diffs = zipWith (-) (toList labels) (V.toList ests)
>   -- putStrLn $ show diffs
>   putStrLn $ show $ (sum (map (^2) diffs) / (fromIntegral $ length diffs))
>

FIXME: Reference for neural net, multi-layer perceptron and logistic
regression.

We can view neural nets or at least a multi layer perceptron as a
generalisation of (multivariate) linear logistic regression. It is
instructive to apply both backpropagation and automated
differentiation to this simpler problem.

Following [Ng][Ng:cs229], we define:

  [Ng:cs229]: http://cs229.stanford.edu

$$
h_{\theta}(\vec{x}) = g(\theta^T\vec{x})
$$

where $\theta = (\theta_1, \ldots, \theta_m)$ and $g$ is a function
such as the logistic function $g(x) = 1 / (1 + e^{-\theta^T\vec{x}})$
or $\tanh$.

Next we define the probability of getting a particular value of the binary lable:

$$
\begin{align*}
{\mathbb P}(y = 1 \mid \vec{x}; \theta) &= h_{\theta}(\vec{x}) \\
{\mathbb P}(y = 0 \mid \vec{x}; \theta) &= 1 - h_{\theta}(\vec{x})
\end{align*}
$$

which we can re-write as:

$$
p(y \mid \vec{x} ; \theta) = (h_{\theta}(\vec{x}))^y(1 - h_{\theta}(\vec{x}))^{1 - y}
$$

We wish to find the value of $\theta$ that gives the maximum
probability to the observations. We do this by maximising the
likelihood. Assuming we have $n$ observations the likelihood is:

$$
\begin{align*}
L(\theta) &= \prod_{i=1}^n p(y^{(i)} \mid {\vec{x}}^{(i)} ; \theta) \\
          &= \prod_{i=1}^n (h_{\theta}(\vec{x}^{(i)}))^{y^{(i)}} (1 - h_{\theta}(\vec{x}^{(i)}))^{1 - y^{(i)}}
\end{align*}
$$

It is standard practice to maximise the log likelihood which will give the same maximum as log is monotonic.

$$
\begin{align*}
l(\theta) &= \log L(\theta) \\
          &= \sum_{i=1}^n {y^{(i)}}\log h_{\theta}(\vec{x}^{(i)}) + (1 - y^{(i)})\log (1 - h_{\theta}(\vec{x}^{(i)}))
\end{align*}
$$

We now use [gradient descent][GradientDescent] to find the maximum by
starting with a random value for the unknown parameter and then
stepping in the steepest direction.

  [GradientDescent]: http://en.wikipedia.org/wiki/Gradient_descent

$$
\theta' = \theta + \gamma\nabla_{\theta}l(\theta)
$$

Differentiating the log likelihood, we have:

$$
\begin{align*}
\frac{\partial}{\partial \theta_i}l(\theta) &= \big(y\frac{1}{g(\theta^T\vec{x})} - (1 - y)\frac{1}{1 - g(\theta^T\vec{x})}\big)\frac{\partial}{\partial \theta_i}g(\theta^T\vec{x})
\end{align*}
$$

> gamma :: Double
> gamma = 0.01
>
> logit :: Floating a => V.Vector a -> V.Vector a -> a
> logit x theta = 1 / (1 + exp (V.sum $ V.zipWith (*) theta x))
>
> logLikelihood :: Floating a => V.Vector a -> a -> V.Vector a -> a
> logLikelihood theta l x = l * log (logit x theta) + (1 - l) * log (1 - logit x theta)
>
> initWs :: V.Vector Double
> initWs = V.replicate 11 0.1
>
> logReg :: IO ()
> logReg = do
>   vals <- withFile "/Users/dom/Downloadable/DataScienceLondon/Train.csv" ReadMode
>           (\h -> do c <- BS.hGetContents h
>                     let mvv :: Either String (V.Vector (V.Vector Double))
>                         mvv = decodeWith myOptions True c
>                     case mvv of
>                       Left s -> do putStrLn s
>                                    return V.empty
>                       Right vv -> return vv
>           )
>   let labels = V.map (V.! 0) vals
>       inds  = V.map (V.drop 1) vals
>       featuress = V.map (V.splitAt 11) inds
>       tF x = log $ x + 1
>       xTrain = V.map (\fs -> V.zipWith (-) (V.map tF $ fst fs) (V.map tF $ snd fs))
>                      featuress
>       delLogLikelihood l x = grad $
>                              \theta -> logLikelihood theta (auto l) (V.map auto x)
>       g theta (l, x) = V.zipWith (+) theta
>                        (V.map (* gamma) $ delLogLikelihood l x theta)
>       bar = V.foldl g initWs (V.zip (V.take 2000 labels) (V.take 2000 xTrain))
>       baz = V.foldl g initWs (V.zip (V.take 2010 labels) (V.take 2010 xTrain))
>   putStrLn $ show $ initWs
>   putStrLn $ show $ bar
>   putStrLn $ show $ V.zipWith (-) bar baz
>   putStrLn $ show $ V.maximum $ V.zipWith (-) bar baz



```{.dia width='400'}
import NnClassifierDia
dia = nn
```
Neural Networks
---------------


We (or rather the authors of the [MonadReader article][MonadReader])
represent an image as a reocord; the pixels are represented using an
8-bit grayscale.

> data Image = Image {
>       iRows    :: Int
>     , iColumns :: Int
>     , iPixels  :: [Word8]
>     } deriving (Eq, Show)

A labelled image contains the image and what this image actually
represents e.g. the image of the numeral 9 could and should be
represented by the value 9.

> type LabelledImage a = ([a], Int)


We follow [@rojas1996neural;@Bishop:2006:PRM:1162264]. We are given a training set:

$$
\{(\vec{x}_0, \vec{y}_0), (\vec{x}_1, \vec{y}_1), \ldots, (\vec{x}_p, \vec{y}_p)\}
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


We are also
given a cost function:

$$
E(\vec{x}) = \frac{1}{2}\sum_1^{N_L}(\hat{y}^L_i - y^L_i)^2
$$

Backpropagation
---------------

As with logistic regression, our goal is to find weights for the
neural network which minimises this cost function. The method that is
used in backpropagation to is to initialise the weights to some small
non-zero amount and then use the method of steepest descent (aka
gradient descent). The idea is that if $f$ is a function of several
variables then to find its minimum value, one ought to take a small
step in the direction in which it is decreasing most quickly and
repeat until no step in any direction results in a decrease. The
analogy is that if one is walking in the mountains then the quickest
way down is to walk in the direction which goes down most steeply. Of
course one get stuck at a local minimum rather than the global minimum
but from a machine learning point of view this may be acceptable;
alternatively one may start at random points in the search space and
check they all give the same minimum.

We therefore need calculate the gradient of the loss function with
respect to the weights (since we need to minimise the cost
function). In other words we need to find:

$$
\nabla E(\vec{x}) \equiv (\frac{\partial E}{\partial x_1}, \ldots, \frac{\partial E}{\partial x_n})
$$

Once we have this we can take our random starting position and move
down the steepest gradient:

$$
w'_i = w_i - \gamma\frac{\partial E}{\partial w_i}
$$

where $\gamma$ is the step length known in machine learning parlance
as the learning rate.


The implementation below is a modified version of [MonadLayer].

We represent a layer as record consisting of the matrix of weights and
the activation function.

> data Layer =
>   Layer
>   {
>     layerWeights  :: Matrix Double,
>     layerFunction :: ActivationFunction
>   }

The activation function itself is a function which takes any type in
the _Floating_ class to the same type in the _Floating_ class e.g. _Double_.

> newtype ActivationFunction =
>   ActivationFunction
>   {
>     activationFunction :: Floating a => a -> a
>   }

Our neural network consists of a list of layers together with a learning rate.

> data BackpropNet = BackpropNet
>     {
>       layers :: [Layer],
>       learningRate :: Double
>     }

The constructor function _buildBackPropnet_ does nothing more than
populate _BackPropNet_ checking that all the matrices of weights are
compatible.  It takes a learning rate, a list of matrices of weights
for each layer, a single common activation function and produce a
neural network.

> buildBackpropNet ::
>   Double ->
>   [Matrix Double] ->
>   ActivationFunction ->
>   BackpropNet
> buildBackpropNet learningRate ws f =
>   BackpropNet {
>       layers       = map buildLayer checkedWeights
>     , learningRate = learningRate
>     }
>   where checkedWeights = scanl1 checkDimensions ws
>         buildLayer w   = Layer { layerWeights  = w
>                                , layerFunction = f
>                                }
>         checkDimensions :: Matrix Double -> Matrix Double -> Matrix Double
>         checkDimensions w1 w2 =
>           if rows w1 == cols w2
>           then w2
>           else error "Inconsistent dimensions in weight matrix"

We keep a record of calculations at each layer in the neural network.

> data PropagatedLayer
>     = PropagatedLayer
>         {
>           propLayerIn         :: ColumnVector Double,
>           propLayerOut        :: ColumnVector Double,
>           propLayerActFun'Val :: ColumnVector Double,
>           propLayerWeights    :: Matrix Double,
>           propLayerActFun     :: ActivationFunction
>         }
>     | PropagatedSensorLayer
>         {
>           propLayerOut :: ColumnVector Double
>         }

We take a record of the calculations at one layer, a layer and produce
the record of the calculations at the next layer.

> propagate :: PropagatedLayer -> Layer -> PropagatedLayer
> propagate layerJ layerK = PropagatedLayer
>         {
>           propLayerIn         = layerJOut,
>           propLayerOut        = mapMatrix f a,
>           propLayerActFun'Val = mapMatrix (diff f) a,
>           propLayerWeights    = weights,
>           propLayerActFun     = layerFunction layerK
>         }
>   where layerJOut = propLayerOut layerJ
>         weights   = layerWeights layerK
>         a         = weights <> layerJOut
>         f :: Floating a => a -> a
>         f = activationFunction $ layerFunction layerK

With this we can take an input to the neural network, the neural
network itself and produce a collection of records of the calculations
at each layer.

> propagateNet :: ColumnVector Double -> BackpropNet -> [PropagatedLayer]
> propagateNet input net = tail calcs
>   where calcs = scanl propagate layer0 (layers net)
>         layer0 = PropagatedSensorLayer $ validateInput net input
>
>         validateInput :: BackpropNet -> ColumnVector Double -> ColumnVector Double
>         validateInput net = validateInputValues . validateInputDimensions net
>
>         validateInputDimensions ::
>           BackpropNet ->
>           ColumnVector Double ->
>           ColumnVector Double
>         validateInputDimensions net input =
>           if got == expected
>           then input
>           else error ("Input pattern has " ++ show got ++ " bits, but " ++
>                       show expected ++ " were expected")
>           where got      = rows input
>                 expected = inputWidth $ head $ layers net
>
>         validateInputValues :: ColumnVector Double -> ColumnVector Double
>         validateInputValues input =
>           if (minimum ns >= 0) && (maximum ns <= 1)
>           then input
>           else error "Input bits outside of range [0,1]"
>           where
>             ns = toList ( flatten input )
>
>         inputWidth :: Layer -> Int
>         inputWidth = cols . layerWeights


We keep a record of the back propagation calculations at each layer in
the neural network:

* The grad of the cost function with respect to the outputs of the layer.
* The grad of the cost function with respect to the weights of the layer.
* The value of the derivative of the activation function.
* The inputs to the layer.
* The outputs from the layer.
* The activation function.

> data BackpropagatedLayer = BackpropagatedLayer
>     {
>       backpropOutGrad    :: ColumnVector Double,
>       backpropWeightGrad :: Matrix Double,
>       backpropActFun'Val :: ColumnVector Double,
>       backpropIn         :: ColumnVector Double,
>       backpropOut        :: ColumnVector Double,
>       backpropWeights    :: Matrix Double,
>       backPropActFun     :: ActivationFunction
>     }

Propagate the inputs backward through this layer to produce an output.

> backpropagate :: PropagatedLayer ->
>                  BackpropagatedLayer ->
>                  BackpropagatedLayer
> backpropagate layerJ layerK = BackpropagatedLayer
>     {
>       backpropOutGrad    = dazzleJ,
>       backpropWeightGrad = errorGrad dazzleJ f'aJ bpIn,
>       backpropActFun'Val = f'aJ,
>       backpropIn         = bpIn,
>       backpropOut        = propLayerOut layerJ,
>       backpropWeights    = propLayerWeights layerJ,
>       backPropActFun     = propLayerActFun layerJ
>     }
>     where dazzleJ = (trans $ backpropWeights layerK) <> (dazzleK * f'aK)
>           dazzleK = backpropOutGrad layerK
>           f'aK    = backpropActFun'Val layerK
>           f'aJ    = propLayerActFun'Val layerJ
>           bpIn    = propLayerIn layerJ

> errorGrad :: ColumnVector Double ->
>              ColumnVector Double ->
>              ColumnVector Double ->
>              Matrix Double
> errorGrad dazzle f'a input = (dazzle * f'a) <> trans input

> backpropagateFinalLayer :: PropagatedLayer ->
>                            ColumnVector Double ->
>                            BackpropagatedLayer
> backpropagateFinalLayer l t = BackpropagatedLayer
>     {
>       backpropOutGrad    = dazzle,
>       backpropWeightGrad = errorGrad dazzle f'a (propLayerIn l),
>       backpropActFun'Val = f'a,
>       backpropIn         = propLayerIn l,
>       backpropOut        = propLayerOut l,
>       backpropWeights    = propLayerWeights l,
>       backPropActFun     = propLayerActFun l
>     }
>     where dazzle =  propLayerOut l - t
>           f'a    = propLayerActFun'Val l

Move backward (from right to left) through the neural network
i.e. this is backpropagation itself.

> backpropagateNet :: ColumnVector Double ->
>                     [PropagatedLayer] ->
>                     [BackpropagatedLayer]
> backpropagateNet target layers = scanr backpropagate layerL hiddenLayers
>   where hiddenLayers = init layers
>         layerL = backpropagateFinalLayer (last layers) target

Now that we know all the derivatives with respect to the weights in
every layer, we can create a new layer by moving one step in the
direction of steepest descent.

> update :: Double ->
>           BackpropagatedLayer ->
>           Layer
> update rate layer = Layer
>         {
>           layerWeights = wNew,
>           layerFunction = backPropActFun layer
>         }
>     where wOld = backpropWeights layer
>           delW = rate `scale` backpropWeightGrad layer
>           wNew = wOld - delW

Now we can train our network by taking a list of inputs and outputs
using each pair to move a step in the direction of steepest descent.

> train :: BackpropNet ->
>          [Double] ->
>          [Double] ->
>          BackpropNet
> train net input target =
>   BackpropNet { layers       = newLayers
>               , learningRate = rate
>               }
>   where newLayers            = map (update $ learningRate net) backpropagatedLayers
>         rate                 = learningRate net
>         backpropagatedLayers = backpropagateNet (listToColumnVector target) propagatedLayers
>         propagatedLayers     = propagateNet x net
>         x                    = listToColumnVector (1:input)

> trainOnePattern :: ([Double], Int) -> BackpropNet -> BackpropNet
> trainOnePattern trainingData net = train net input target
>   where input = fst trainingData
>         digit = snd trainingData
>         target = targets !! digit
>
> targets :: Floating a => [[a]]
> targets = map row [0 .. nDigits - 1]
>   where
>     row m = concat [x, 1.0 : y]
>       where
>         (x, y) = splitAt m (take (nDigits - 1) $ repeat 0.0)

> trainWithAllPatterns :: BackpropNet ->
>                         [([Double], Int)]
>                         -> BackpropNet
> trainWithAllPatterns = foldl' (flip trainOnePattern)

Automated Differentation
------------------------

> data PropagatedLayer' a
>     = PropagatedLayer'
>         {
>           propLayerIn'         :: [a],
>           propLayerOut'        :: [a],
>           propLayerActFun'Val' :: [a],
>           propLayerWeights'    :: [[a]],
>           propLayerActFun'     :: ActivationFunction
>         }
>     | PropagatedSensorLayer'
>         {
>           propLayerOut' :: [a]
>         } deriving (Functor, Foldable, Traversable)

> data Layer' a =
>   Layer'
>   {
>     layerWeights'  :: [[a]],
>     layerFunction' :: ActivationFunction
>   } deriving (Functor, Foldable, Traversable)

> data BackpropNet' a = BackpropNet'
>     {
>       layers'       :: [Layer' a],
>       learningRate' :: Double
>     } deriving (Functor, Foldable, Traversable)

> matMult :: Num a => [[a]] -> [a] -> [a]
> matMult m v = map (\r -> sum $ zipWith (*) r v) m

> propagate' :: Floating a => PropagatedLayer' a -> Layer' a -> PropagatedLayer' a
> propagate' layerJ layerK = PropagatedLayer'
>         {
>           propLayerIn'         = layerJOut,
>           propLayerOut'        = map f a,
>           propLayerActFun'Val' = map (diff f) a,
>           propLayerWeights'    = weights,
>           propLayerActFun'     = layerFunction' layerK
>         }
>   where layerJOut = propLayerOut' layerJ
>         weights   = layerWeights' layerK
>         a = weights `matMult` layerJOut
>         f :: Floating a => a -> a
>         f = activationFunction $ layerFunction' layerK

> propagateNet' :: (Floating a, Ord a) => [a] -> BackpropNet' a -> [PropagatedLayer' a]
> propagateNet' input net = tail calcs
>   where calcs = scanl propagate' layer0 (layers' net)
>         layer0 = PropagatedSensorLayer' $ validateInput net input
>
>         validateInput net = validateInputValues . validateInputDimensions net
>
>         validateInputDimensions net input =
>           if got == expected
>           then input
>           else error ("Input pattern has " ++ show got ++ " bits, but " ++
>                       show expected ++ " were expected")
>           where got      = length input
>                 expected = length $ head $ layerWeights' $ head $ layers' net
>
>         validateInputValues input =
>           if (minimum input >= 0) && (maximum input <= 1)
>           then input
>           else error "Input bits outside of range [0,1]"

> buildBackpropNet' ::
>   Double ->
>   [[[a]]] ->
>   ActivationFunction ->
>   BackpropNet' a
> buildBackpropNet' learningRate ws f =
>   BackpropNet' {
>       layers'       = map buildLayer checkedWeights
>     , learningRate' = learningRate
>     }
>   where checkedWeights = scanl1 checkDimensions ws
>         buildLayer w   = Layer' { layerWeights'  = w
>                                 , layerFunction' = f
>                                 }
>         checkDimensions :: [[a]] -> [[a]] -> [[a]]
>         checkDimensions w1 w2 =
>           if length w1 == length (head w2)
>           then w2
>           else error $ "Inconsistent dimensions in weight matrix\n" ++
>                         show (length w1)        ++ "\n" ++
>                         show (length w2)        ++ "\n" ++
>                         show (length $ head w1) ++ "\n" ++
>                         show (length $ head w2)

> evaluateBPN' :: (Floating a, Ord a) => BackpropNet' a -> [a] -> [a]
> evaluateBPN' net input = propLayerOut' $ last calcs
>   where calcs = propagateNet' (1:input) net
>
> costFn :: (Floating a, Ord a) => Int -> [a] -> BackpropNet' a -> a
> costFn expectedDigit input net = 0.5 * sum (map (^2) diffs)
>   where
>     predicted = evaluateBPN' net input
>     diffs = zipWith (-) (targets!!expectedDigit) predicted

> evaluateBPN :: BackpropNet -> [Double] -> [Double]
> evaluateBPN net input = columnVectorToList $ propLayerOut $ last calcs
>   where calcs = propagateNet x net
>         x = listToColumnVector (1:input)

> evalOnePattern :: BackpropNet -> ([Double], Int) -> Int
> evalOnePattern net trainingData =
>   isMatch result target
>   where input = fst trainingData
>         target = snd trainingData
>         rawResult = evaluateBPN net input
>         result = interpret rawResult

> evalAllPatterns :: BackpropNet -> [([Double], Int)] -> [Int]
> evalAllPatterns = map . evalOnePattern


Appendix
--------

In order to run the trained neural network then we need some training
data and test data.

FIXME: We can probably get the number of rows and columns from the
data itself.

Our neural net configuration. We wish to classify images which are $28
\times 28$ pixels into 10 digits using a single layer neural net with
20 nodes.

> lRate :: Double
> lRate = 0.007
> nRows, nCols, nNodes, nDigits :: Int
> nRows = 28
> nCols = 28
> nNodes = 20
> nDigits = 10
>
> smallRandoms :: (Random a, Floating a) => Int -> [a]
> smallRandoms seed = map (/100) (randoms (mkStdGen seed))
>
> randomWeightMatrix :: Int -> Int -> Int -> Matrix Double
> randomWeightMatrix numInputs numOutputs seed = x
>   where
>     x = (numOutputs >< numInputs) weights
>     weights = take (numOutputs * numInputs) (smallRandoms seed)
>
> randomWeightMatrix' :: (Floating a, Random a) => Int -> Int -> Int -> [[a]]
> randomWeightMatrix' numInputs numOutputs seed = y
>   where
>     -- y :: (Random a, Floating a) => [[a]]
>     y = chunksOf numInputs weights
>     -- weights :: (Random a, Floating a) => [a]
>     weights = take (numOutputs * numInputs) (smallRandoms seed)

> main :: IO ()
> main = do
>   linReg
>   logReg
>   let w1  = randomWeightMatrix (nRows * nCols + 1) nNodes 7
>       w2  = randomWeightMatrix nNodes nDigits 42
>       w1' :: (Random a, Floating a) => [[a]]
>       w1' = randomWeightMatrix' (nRows * nCols + 1) nNodes 7
>       w2' :: (Random a, Floating a) => [[a]]
>       w2' = randomWeightMatrix' nNodes nDigits 42
>       initialNet  = buildBackpropNet  lRate [w1, w2] tanhAS
>       initialNet' :: BackpropNet' Double
>       initialNet' = buildBackpropNet' lRate [w1', w2'] tanhAS
>
>   trainingData <- fmap (take 8000) readTrainingData
>
>   trainingLabels <- readLabels "train-labels-idx1-ubyte"
>   trainingImages <- readImages "train-images-idx3-ubyte"
>
>   let trainingData' :: RealFrac a => [LabelledImage a]
>       trainingData' = zip (map normalisedData' trainingImages) trainingLabels
>
>   let costWeights :: (Ord a, RealFrac a, Floating a) => BackpropNet' a -> a
>       costWeights = costFn (snd $ head trainingData') (fst $ head trainingData')
>       derivCostWeights :: (Floating a, RealFrac a) => BackpropNet' a -> BackpropNet' a
>       derivCostWeights = grad costWeights
>
>   error $ take 1000 $ show $ map layerWeights' $ layers' $ derivCostWeights initialNet'
>
>   let finalNet = trainWithAllPatterns initialNet trainingData
>
>   testData <- fmap (take 1000) readTestData
>   putStrLn $ "Testing with " ++ show (length testData) ++ " images"
>   let results = evalAllPatterns finalNet testData
>   let score = fromIntegral (sum results)
>   let count = fromIntegral (length testData)
>   let percentage = 100.0 * score / count
>   putStrLn $ "I got " ++ show percentage ++ "% correct"

> deserialiseLabels :: Get (Word32, Word32, [Word8])
> deserialiseLabels = do
>   magicNumber <- getWord32be
>   count <- getWord32be
>   labelData <- getRemainingLazyByteString
>   let labels = BL.unpack labelData
>   return (magicNumber, count, labels)
>
> readLabels :: FilePath -> IO [Int]
> readLabels filename = do
>   content <- BL.readFile filename
>   let (_, _, labels) = runGet deserialiseLabels content
>   return (map fromIntegral labels)
>

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
>
> readImages :: FilePath -> IO [Image]
> readImages filename = do
>   content <- BL.readFile filename
>   let (_, _, r, c, unpackedData) = runGet deserialiseHeader content
>   return (map (Image (fromIntegral r) (fromIntegral c)) unpackedData)
>
> -- | Inputs, outputs and targets are represented as column vectors instead of lists
> type ColumnVector a = Matrix a
>
>  -- | Convert a column vector to a list
> columnVectorToList :: (Ord a, Field a)
>     -- | The column vector to convert
>     => ColumnVector a
>     -- | The resulting list
>     -> [a]
> columnVectorToList = toList . flatten
>
> -- | Convert a list to a column vector
> listToColumnVector :: (Ord a, Field a)
>     -- | the list to convert
>     => [a]
>     -- | the resulting column vector
>     -> ColumnVector a
> listToColumnVector x = (len >< 1) x
>     where len = length x

> tanhAS :: ActivationFunction
> tanhAS = ActivationFunction
>     {
>       activationFunction = tanh
>     }

FIXME: This looks a bit yuk

> isMatch :: (Eq a) => a -> a -> Int
> isMatch x y =
>   if x == y
>   then 1
>   else 0

> interpret :: [Double] -> Int
> interpret v = fromJust (elemIndex (maximum v) v)

> readTrainingData ::  Floating a => IO [LabelledImage a]
> readTrainingData = do
>   trainingLabels <- readLabels "train-labels-idx1-ubyte"
>   trainingImages <- readImages "train-images-idx3-ubyte"
>   return $ zip (map normalisedData trainingImages) trainingLabels

> readTrainingData' ::  RealFrac a => IO [LabelledImage a]
> readTrainingData' = do
>   trainingLabels <- readLabels "train-labels-idx1-ubyte"
>   trainingImages <- readImages "train-images-idx3-ubyte"
>   return $ zip (map normalisedData' trainingImages) trainingLabels

> readTestData :: Floating a => IO [LabelledImage a]
> readTestData = do
>   putStrLn "Reading test labels..."
>   testLabels <- readLabels "t10k-labels-idx1-ubyte"
>   testImages <- readImages "t10k-images-idx3-ubyte"
>   return (zip (map normalisedData testImages) testLabels)

> normalisedData :: Floating a => Image -> [a]
> normalisedData image = map normalisePixel (iPixels image)
>   where
>     normalisePixel :: Floating a => Word8 -> a
>     normalisePixel p = (fromIntegral p) / 255.0

> normalisedData' :: RealFrac a => Image -> [a]
> normalisedData' image = map normalisePixel (iPixels image)
>   where
>     normalisePixel p = (fromIntegral p) / 255.0

