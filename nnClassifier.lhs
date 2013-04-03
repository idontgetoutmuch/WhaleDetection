The article "A Functional Approach to Neural Networks" in the [Monad
Reader][MonadReader] shows how to use a Neural Network to classify
handwritten digits in the [MNIST database][MNIST].

  [MonadReader]: http://themonadreader.files.wordpress.com/2013/03/issue21.pdf
  [MNIST]: http://yann.lecun.com/exdb/mnist/
  [LeCunCortesMnist]: http://yann.lecun.com/exdb/mnist/

FIXME: More references required.

The reader is struck by how similar [backpropogation][Backpropogation]
is to [automatic differentiation][AutomaticDifferentiation]. The
reader may not therefore be surprised to find that this observation
had been made before: [Domke2009a][Domke2009a]. Indeed as Dan Piponi
observes: "the grandaddy machine-learning algorithm of them all,
back-propagation, is nothing but steepest descent with reverse mode
automatic differentiation".

  [Backpropogation]: http://en.wikipedia.org/wiki/Backpropagation
  [AutomaticDifferentiation]: http://en.wikipedia.org/wiki/Automatic_differentiation
  [Domke2009a]: http://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/

This article is divided into two parts: the first is summarises how
backpropogation works and the second shows how this can be replaced by
automatic differentation. Both techniques are applied to what appears
to be the standard [benchmark][LeCunCortesMnist]. In the first part,
we follow the exposition given in the [MonadReader][MonadReader].

Multivariate Linear Logistic Regression
---------------------------------------

FIXME: Reference for neural net, multi-layer perceptron and logistic
regression.

We can view neural nets or at least a multi layer perceptron as a
generalisation of (multivariate) linear logistic regression. It is
instructive to apply both backpropogation and automated
differentiation to this simpler problem.

Backpropogation
---------------


> import MarineExplore
> import Runner
> import Backprop

> import Numeric.LinearAlgebra
> import Data.List
> import System.Random

> import Numeric.AD

> learningRate = 0.007
> nRows, nCols :: Int
> nRows = 9 -- 129
> nCols = 9 -- 49

> smallRandoms :: Int -> [Double]
> smallRandoms seed = map (/100) (randoms (mkStdGen seed))

> randomWeightMatrix :: Int -> Int -> Int -> Matrix Double
> randomWeightMatrix numInputs numOutputs seed = (numOutputs><numInputs) weights
>     where weights = take (numOutputs*numInputs) (smallRandoms seed)

> zeroWeightMatrix :: Int -> Int -> Matrix Double
> zeroWeightMatrix numInputs numOutputs = (numOutputs><numInputs) weights
>     where weights = repeat 0

> main :: IO ()
> main = do
>   let w1 = randomWeightMatrix (nRows * nCols + 1) 4 7
>   let w2 = randomWeightMatrix 4 2 42
>   let initialNet = buildBackpropNet learningRate [w1, w2] tanhAS
>   trainingData2 <- readTrainingData

-- >   let finalNet = trainWithAllPatterns initialNet (take 1800 trainingData2)

>   let finalNet = trainWithAllPatterns initialNet (take 93 trainingData2)
>   testData2 <- readTestData

-- >   let results = evalAllPatterns finalNet (take 200 $ drop 1800 testData2)

>   let results = evalAllPatterns finalNet (take 7 $ drop 93 testData2)

-- >   let expects = map snd $ take 200 $ drop 1800 testData2

>   let expects = map snd $ take 7 $ drop 93 testData2
>   putStrLn $ "Score:\n" ++ show results
>   putStrLn $ "Expected:\n" ++ show expects
