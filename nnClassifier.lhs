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

FIXME: For exposition we might have to inline all the modules.

> import Mnist
> import Runner
> import Backprop
>
> import Numeric.LinearAlgebra
> import Numeric.AD
> import Data.List
> import Data.List.Split
> import Data.Foldable hiding (sum)
> import System.Random

FIXME: Explain the learning rate and initialisation.

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
> smallRandoms :: Int -> [Double]
> smallRandoms seed = map (/100) (randoms (mkStdGen seed))
>
> randomWeightMatrix :: Int -> Int -> Int -> Matrix Double
> randomWeightMatrix numInputs numOutputs seed = (numOutputs><numInputs) weights
>     where weights = take (numOutputs*numInputs) (smallRandoms seed)
>
> zeroWeightMatrix :: Int -> Int -> Matrix Double
> zeroWeightMatrix numInputs numOutputs = (numOutputs><numInputs) weights
>     where weights = repeat 0

FIXME: Fix forcing to use something other than putStrLn.

> myForce :: BackpropNet -> [LabelledImage] -> IO BackpropNet
> myForce oldNet trainingData = do
>   let newNet = trainWithAllPatterns oldNet trainingData
>   putStrLn $ show $ length $ toLists $ lW $ head $ layers newNet
>   putStrLn $ show $ length $ head $ toLists $ lW $ head $ layers newNet
>   return newNet
>
> update :: (Integer, Integer) -> BackpropNet -> IO BackpropNet
> update (start, end) oldNet = do
>   allTrainingData <- readTrainingData start end
>   myForce oldNet allTrainingData
>
> main :: IO ()
> main = do
>   let w1 = randomWeightMatrix (nRows * nCols + 1) nNodes 7
>   let w2 = randomWeightMatrix nNodes nDigits 42
>   let initialNet = buildBackpropNet lRate [w1, w2] tanhAS
>
>   finalNet <- foldrM update initialNet [ (   0,    999), (1000,   1999)
>                                        , (2000,   2999), (3000,   3999)
>                                        , (4000,   4999), (5000,   5999)
>                                        , (6000,   6999), (7000,   7999)
>                                        , (8000,   8999), (9000,   9999)
>                                        , (10000, 10999), (11000, 11999)
>                                        , (12000, 12999), (13000, 13999)
>                                        , (14000, 14999), (15000, 15999)
>                                        , (16000, 16999), (17000, 17999)
>                                        , (18000, 18999), (19000, 19999)
>                                        , (20000, 20999), (21000, 21999)
>                                        , (22000, 22999), (23000, 23999)
>                                        , (24000, 24999), (25000, 25999)
>                                        , (26000, 26999), (27000, 27999)
>                                        ]
>
>   testData2 <- readTestData
>   let testData = take 1000 testData2
>   putStrLn $ "Testing with " ++ show (length testData) ++ " images"
>   let results = evalAllPatterns finalNet testData
>   let score = fromIntegral (sum results)
>   let count = fromIntegral (length testData)
>   let percentage = 100.0 * score / count
>   putStrLn $ "I got " ++ show percentage ++ "% correct"
