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

> {- OPTIONS_GHC -Wall                    #-}
> {- OPTIONS_GHC -fno-warn-name-shadowing #-}
> {- OPTIONS_GHC -fno-warn-type-defaults  #-}

> import Numeric.LinearAlgebra
> import Numeric.AD
> import Data.List
> import Data.List.Split
> import Data.Foldable (foldrM)
> import System.Random

For use in the appendix.

> import Data.Word
> import qualified Data.ByteString.Lazy as BL
> import Data.Binary.Get
> import Data.Binary.Put
>
> import Debug.Trace
> import Data.Maybe

We (or rather the authors of the [MonadReader article][MonadReader])
represent an image as a reocord; the pixels are represented using an
8-bit grayscale.

> data Image = Image {
>       iRows    :: Int
>     , iColumns :: Int
>     , iPixels  :: [Word8]
>     } deriving (Eq, Show)
>
> toMatrix :: Image -> Matrix Double
> toMatrix image = (r><c) p :: Matrix Double
>   where r = iRows image
>         c = iColumns image
>         p = map fromIntegral (iPixels image)
>
> type LabelledImage = ([Double], Int)


> normalisedData :: Image -> [Double]
> normalisedData image = map normalisePixel (iPixels image)
>   where
>     normalisePixel :: Word8 -> Double
>     normalisePixel p = (fromIntegral p) / 255.0

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

The meat of the article yet to be explained.

> data ActivationSpec = ActivationSpec
>     {
>       asF :: Double -> Double,
>       asF' :: Double -> Double,
>       desc :: String
>     }

FIXME: This is pretty rather than show.

> instance Show ActivationSpec where
>   show = desc

> -- | An individual layer in a neural network, after propagation but prior to backpropagation
> data PropagatedLayer
>     = PropagatedLayer
>         {
>           -- The input to this layer
>           pIn :: ColumnVector Double,
>           -- The output from this layer
>           pOut :: ColumnVector Double,
>           -- The value of the first derivative of the activation function for this layer
>           pF'a :: ColumnVector Double,
>           -- The weights for this layer
>           pW :: Matrix Double,
>           -- The activation specification for this layer
>           pAS :: ActivationSpec
>         }
>     | PropagatedSensorLayer
>         {
>           -- The output from this layer
>           pOut :: ColumnVector Double
>         }

> propagateNet :: ColumnVector Double -> BackpropNet -> [PropagatedLayer]
> propagateNet input net = tail calcs
>   where calcs = scanl propagate layer0 (layers net)
>         layer0 = PropagatedSensorLayer{ pOut=validatedInputs }
>         validatedInputs = validateInput net input

> -- | Propagate the inputs through this layer to produce an output.
> propagate :: PropagatedLayer -> Layer -> PropagatedLayer
> propagate layerJ layerK = PropagatedLayer
>         {
>           pIn = x,
>           pOut = y,
>           pF'a = f'a,
>           pW = w,
>           pAS = lAS layerK
>         }
>   where x = pOut layerJ
>         w = lW layerK
>         a = w <> x
>         f = asF ( lAS layerK )
>         y = mapMatrix f a
>         f' = asF' ( lAS layerK )
>         f'a = mapMatrix f' a

> validateInput :: BackpropNet -> ColumnVector Double -> ColumnVector Double
> validateInput net = validateInputValues . validateInputDimensions net
>
> validateInputDimensions ::
>     BackpropNet ->
>     ColumnVector Double ->
>     ColumnVector Double
> validateInputDimensions net input =
>   if got == expected
>        then input
>        else error ("Input pattern has " ++ show got ++ " bits, but " ++ show expected ++ " were expected")
>            where got = rows input
>                  expected = inputWidth (head (layers net))
>
> validateInputValues :: ColumnVector Double -> ColumnVector Double
> validateInputValues input =
>   if (min >= 0) && (max <= 1)
>        then input
>        else error "Input bits outside of range [0,1]"
>        where min = minimum ns
>              max = maximum ns
>              ns = toList ( flatten input )

> inputWidth :: Layer -> Int
> inputWidth = cols . lW


> -- | An individual layer in a neural network, after backpropagation
> data BackpropagatedLayer = BackpropagatedLayer
>     {
>       -- Del-sub-z-sub-l of E
>       bpDazzle :: ColumnVector Double,
>       -- The error due to this layer
>       bpErrGrad :: ColumnVector Double,
>       -- The value of the first derivative of the activation
>       --   function for this layer
>       bpF'a :: ColumnVector Double,
>       -- The input to this layer
>       bpIn :: ColumnVector Double,
>       -- The output from this layer
>       bpOut :: ColumnVector Double,
>       -- The weights for this layer
>       bpW :: Matrix Double,
>       -- The activation specification for this layer
>       bpAS :: ActivationSpec
>     }

> backpropagateNet ::
>   ColumnVector Double -> [PropagatedLayer] -> [BackpropagatedLayer]
> backpropagateNet target layers = scanr backpropagate layerL hiddenLayers
>   where hiddenLayers = init layers
>         layerL = backpropagateFinalLayer (last layers) target

> backpropagateFinalLayer ::
>     PropagatedLayer -> ColumnVector Double -> BackpropagatedLayer
> backpropagateFinalLayer l t = BackpropagatedLayer
>     {
>       bpDazzle = dazzle,
>       bpErrGrad = errorGrad dazzle f'a (pIn l),
>       bpF'a = pF'a l,
>       bpIn = pIn l,
>       bpOut = pOut l,
>       bpW = pW l,
>       bpAS = pAS l
>     }
>     where dazzle =  pOut l - t
>           f'a = pF'a l

> errorGrad :: ColumnVector Double -> ColumnVector Double -> ColumnVector Double
>     -> ColumnVector Double
> errorGrad dazzle f'a input = (dazzle * f'a) <> trans input

> -- | An individual layer in a neural network, prior to propagation
> data Layer = Layer
>     {
>       -- The weights for this layer
>       lW :: Matrix Double,
>       -- The activation specification for this layer
>       lAS :: ActivationSpec
>     } deriving Show

> data BackpropNet = BackpropNet
>     {
>       layers :: [Layer],
>       learningRate :: Double
>     } deriving Show

> buildBackpropNet ::
>   -- The learning rate
>   Double ->
>   -- The weights for each layer
>   [Matrix Double] ->
>   -- The activation specification (used for all layers)
>   ActivationSpec ->
>   -- The network
>   BackpropNet
> buildBackpropNet lr ws s = BackpropNet { layers=ls, learningRate=lr }
>   where checkedWeights = scanl1 checkDimensions ws
>         ls = map buildLayer checkedWeights
>         buildLayer w = Layer { lW=w, lAS=s }

> -- | Propagate the inputs backward through this layer to produce an output.
> backpropagate :: PropagatedLayer -> BackpropagatedLayer -> BackpropagatedLayer
> backpropagate layerJ layerK = BackpropagatedLayer
>     {
>       bpDazzle = dazzleJ,
>       bpErrGrad = errorGrad dazzleJ f'aJ (pIn layerJ),
>       bpF'a = pF'a layerJ,
>       bpIn = pIn layerJ,
>       bpOut = pOut layerJ,
>       bpW = pW layerJ,
>       bpAS = pAS layerJ
>     }
>     where dazzleJ = wKT <> (dazzleK * f'aK)
>           dazzleK = bpDazzle layerK
>           wKT = trans ( bpW layerK )
>           f'aK = bpF'a layerK
>           f'aJ = pF'a layerJ

> update :: Double -> BackpropagatedLayer -> Layer
> update rate layer = Layer
>         {
>           lW = wNew,
>           lAS = bpAS layer
>         }
>     where wOld = bpW layer
>           delW = rate `scale` bpErrGrad layer
>           wNew = wOld - delW

> evaluateBPN :: BackpropNet -> [Double] -> [Double]
> evaluateBPN net input = columnVectorToList( pOut ( last calcs ))
>   where calcs = propagateNet x net
>         x = listToColumnVector (1:input)
>
> trainBPN :: BackpropNet -> [Double] -> [Double] -> BackpropNet
> trainBPN net input target = BackpropNet { layers=newLayers, learningRate=rate }
>   where newLayers = map (update rate) backpropagatedLayers
>         rate = learningRate net
>         backpropagatedLayers = backpropagateNet (listToColumnVector target) propagatedLayers
>         propagatedLayers = propagateNet x net
>         x = listToColumnVector (1:input)

> evalOnePattern net trainingData =
>   trace (show target ++ ":" ++ show rawResult ++ ":" ++ show result ++ ":" ++ show ((rawResult!!1) / (rawResult!!0))) $
>   isMatch result target
>   where input = fst trainingData
>         target = snd trainingData
>         rawResult = evaluateBPN net input
>         result = interpret rawResult

> evalAllPatterns = map . evalOnePattern

> trainOnePattern trainingData net = trainBPN net input target
>   where input = fst trainingData
>         digit = snd trainingData
>         target = targets !! digit

> trainWithAllPatterns = foldl' (flip trainOnePattern)

FIXME: Fix forcing to use something other than putStrLn.

> myForce :: BackpropNet -> [LabelledImage] -> IO BackpropNet
> myForce oldNet trainingData = do
>   let newNet = trainWithAllPatterns oldNet trainingData
>   putStrLn $ show $ length $ toLists $ lW $ head $ layers newNet
>   putStrLn $ show $ length $ head $ toLists $ lW $ head $ layers newNet
>   return newNet
>
> update' :: (Integer, Integer) -> BackpropNet -> IO BackpropNet
> update' (start, end) oldNet = do
>   allTrainingData <- readTrainingData start end
>   myForce oldNet allTrainingData
>
> main :: IO ()
> main = do
>   let w1 = randomWeightMatrix (nRows * nCols + 1) nNodes 7
>   let w2 = randomWeightMatrix nNodes nDigits 42
>   let initialNet = buildBackpropNet lRate [w1, w2] tanhAS
>
>   finalNet <- foldrM update' initialNet [ (   0,    999), (1000,   1999)
>                                         , (2000,   2999), (3000,   3999)
>                                         , (4000,   4999), (5000,   5999)
>                                         , (6000,   6999), (7000,   7999)
>                                         , (8000,   8999), (9000,   9999)
>                                         , (10000, 10999), (11000, 11999)
>                                         , (12000, 12999), (13000, 13999)
>                                         , (14000, 14999), (15000, 15999)
>                                         , (16000, 16999), (17000, 17999)
>                                         , (18000, 18999), (19000, 19999)
>                                         , (20000, 20999), (21000, 21999)
>                                         , (22000, 22999), (23000, 23999)
>                                         , (24000, 24999), (25000, 25999)
>                                         , (26000, 26999), (27000, 27999)
>                                         ]
>
>   testData2 <- readTestData
>   let testData = take 1000 testData2
>   putStrLn $ "Testing with " ++ show (length testData) ++ " images"
>   let results = evalAllPatterns finalNet testData
>   let score = fromIntegral (sum results)
>   let count = fromIntegral (length testData)
>   let percentage = 100.0 * score / count
>   putStrLn $ "I got " ++ show percentage ++ "% correct"

Appendix
--------

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
> serialiseLabels :: Word32 -> Word32 -> [Word8] -> Put
> serialiseLabels magicNumber count labels = do
>   putWord32be magicNumber
>   putWord32be count
>   mapM_ putWord8 labels
>
> writeLabels :: FilePath -> [Int] -> IO ()
> writeLabels fileName labels = do
>   let content = runPut $ serialiseLabels
>                          0x00000801
>                          (fromIntegral $ length labels)
>                          (map fromIntegral labels)
>   BL.writeFile fileName content


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
> deserialiseHeader' :: Integer -> Integer -> Get (Word32, Word32, Word32, Word32, [[Word8]])
> deserialiseHeader' start end = do
>   magicNumber <- getWord32be
>   imageCount <- getWord32be
>   r <- getWord32be
>   c <- getWord32be
>   let len = fromIntegral (r * c)
>   _ <- getLazyByteString (fromIntegral $ len * start)
>   packedData <- getLazyByteString (fromIntegral $ len * (end - start + 1))
>   let unpackedData = chunksOf (fromIntegral len) (BL.unpack packedData)
>   return (magicNumber, imageCount, r, c, unpackedData)
>
> readImages :: FilePath -> IO [Image]
> readImages filename = do
>   content <- BL.readFile filename
>   let (_, _, r, c, unpackedData) = runGet deserialiseHeader content
>   return (map (Image (fromIntegral r) (fromIntegral c)) unpackedData)
>
> readImages' :: FilePath -> Integer -> Integer -> IO [Image]
> readImages' filename start end = do
>   content <- BL.readFile filename
>   let (_, _, r, c, unpackedData) = runGet (deserialiseHeader' start end) content
>   return (map (Image (fromIntegral r) (fromIntegral c)) unpackedData)
>
> serialiseHeader :: Word32 -> Word32 -> Word32 -> Word32 -> [[Word8]] -> Put
> serialiseHeader magicNumber imageCount nRows nCols iss = do
>   putWord32be magicNumber
>   putWord32be imageCount
>   putWord32be nRows
>   putWord32be nCols
>   mapM_ putWord8 $ concat iss
>
> writeImages :: FilePath -> [Image] -> IO ()
> writeImages fileName is = do
>   let content = runPut $ serialiseHeader
>                          0x00000803
>                          (fromIntegral $ length is)
>                          (fromIntegral $ iRows $ head is)
>                          (fromIntegral $ iColumns $ head is)
>                          (map iPixels is)
>   BL.writeFile fileName content
>
> writeImage :: FilePath -> Image -> IO ()
> writeImage fileName i = do
>  let content = runPut $ mapM_ putWord8 $ iPixels i
>  BL.appendFile fileName content
>
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


> logisticSigmoid :: (Field a, Floating a) => a -> a -> a
> logisticSigmoid c a = 1 / (1 + exp((-c) * a))
>
> logisticSigmoid' :: (Field a, Floating a) => a -> a -> a
> logisticSigmoid' c a = (c * f a) * (1 - f a)
>   where f = logisticSigmoid c
>
>
> tanhAS :: ActivationSpec
> tanhAS = ActivationSpec
>     {
>       asF = tanh,
>       asF' = tanh',
>       desc = "tanh"
>     }
>
> tanh' x = 1 - (tanh x)^2

> checkDimensions :: Matrix Double -> Matrix Double -> Matrix Double
> checkDimensions w1 w2 =
>   if rows w1 == cols w2
>        then w2
>        else error "Inconsistent dimensions in weight matrix"

FIXME: Hem hem surely we can generate this automatically

> targets :: [[Double]]
> targets =
>     [
>         [0.9, 0.1]
>       , [0.1, 0.9]
>     ]

FIXME: This looks a bit yuk

> isMatch :: (Eq a) => a -> a -> Int
> isMatch x y =
>   if x == y
>   then 1
>   else 0

> interpret :: [Double] -> Int
> interpret v = fromJust (elemIndex (maximum v) v)

> readTrainingData :: Integer -> Integer -> IO [LabelledImage]
> readTrainingData start end = do
>   trainingLabels <- readLabels "whales-labels-test.mnist"
>   trainingImages <- readImages' "pca-images-train.mnist" start end
>   return $ {- enrich $ -} zip (map normalisedData trainingImages) trainingLabels
>
> readTestData :: IO [LabelledImage]
> readTestData = do
>   putStrLn "Reading test labels..."
>   testLabels <- readLabels "whales-labels-test.mnist"
>   testImages <- readImages "pca-images-train.mnist"
>   return (zip (map normalisedData testImages) testLabels)
