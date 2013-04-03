module Runner (
    readTrainingData
  , readTestData
  , trainWithAllPatterns
  , evalAllPatterns
  , LabelledImage(..)
  ) where

import NeuralNet
import Mnist
import Backprop

import Numeric.LinearAlgebra
import Data.List
import Data.Maybe
import System.Random
import Debug.Trace



targets :: [[Double]]
targets =
    [
        [0.9, 0.1]
      , [0.1, 0.9]
    ]

-- targets :: [[Double]]
-- targets =
--     [
--         [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
--       , [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
--       , [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
--       , [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
--       , [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
--       , [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
--       , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
--       , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
--       , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1]
--       , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
--     ]


{-
targets :: [[Double]]
targets =
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      , [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      , [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      , [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      , [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      , [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
      , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
      , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
      , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
      , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
-}

interpret :: [Double] -> Int
interpret v = fromJust (elemIndex (maximum v) v)

isMatch :: (Eq a) => a -> a -> Int
isMatch x y =
  if x == y
  then 1
  else 0

type LabelledImage = ([Double], Int)
-- ^ The training set, an array of input/target pairs

trainOnePattern
  :: (NeuralNet n)
  => LabelledImage
  -> n
  -> n
trainOnePattern trainingData net = train net input target
  where input = fst trainingData
        digit = snd trainingData
        target = targets !! digit

trainWithAllPatterns
  :: (NeuralNet n)
  => n
  -> [LabelledImage]
  -> n
trainWithAllPatterns = foldl' (flip trainOnePattern)

evalOnePattern
  :: (NeuralNet n)
  => n
  -> LabelledImage
  -> Int
evalOnePattern net trainingData =
  trace (show target ++ ":" ++ show rawResult ++ ":" ++ show result ++ ":" ++ show ((rawResult!!1) / (rawResult!!0))) $
  isMatch result target
  where input = fst trainingData
        target = snd trainingData
        rawResult = evaluate net input
        result = interpret rawResult

evalAllPatterns
  :: (NeuralNet n)
  => n
  -> [LabelledImage]
  -> [Int]
evalAllPatterns = map . evalOnePattern

enrich :: [LabelledImage] -> [LabelledImage]
enrich = concat . unfoldr g
  where
    g :: [LabelledImage] -> Maybe ([LabelledImage], [LabelledImage])
    g [] = Nothing
    g ((im, l) : lis) | l == 1    = Just (replicate 4 (im, l), lis)
                      | l == 0    = Just (replicate 1 (im, l), lis)
                      | otherwise = error $ "Unexpected label: " ++ show l

readTrainingData :: Integer -> Integer -> IO [LabelledImage]
readTrainingData start end = do
  trainingLabels <- readLabels "whales-labels-test.mnist"
  trainingImages <- readImages' "pca-images-train.mnist" start end
  return $ {- enrich $ -} zip (map normalisedData trainingImages) trainingLabels

readTestData :: IO [LabelledImage]
readTestData = do
  putStrLn "Reading test labels..."
--  testLabels <- readLabels "t10k-labels-idx1-ubyte"
--  testLabels <- readLabels "vert-or-horiz-labels-test.mnist"
  testLabels <- readLabels "whales-labels-test.mnist"
--  putStrLn $ "Read " ++ show (length testLabels) ++ " labels"
--  putStrLn "Reading test images..."
--  testImages <- readImages "t10k-images-idx3-ubyte"
--  testImages <- readImages "vert-or-horiz-images-test.mnist"
  testImages <- readImages "pca-images-train.mnist"
--  putStrLn $ "Read " ++ show (length testImages) ++ " images"
--  putStrLn "Testing..."
  return (zip (map normalisedData testImages) testLabels)

