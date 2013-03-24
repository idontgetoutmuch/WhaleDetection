{-# OPTIONS_GHC -Wall                    #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing #-}
{-# OPTIONS_GHC -fno-warn-type-defaults  #-}

module Runner
  ( readLabels
  , readTrainingData
  , trainWithAllPatterns
  , readTestData
  , evalAllPatterns
  ) where

import NeuralNet
import MarineExplore

import Data.List
import Data.Maybe

targets :: [[Double]]
targets =
    [
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
      , [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
      , [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
      , [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
      , [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
      , [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
      , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
      , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
      , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1]
      , [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    ]

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
trainWithAllPatterns = foldr trainOnePattern

evalOnePattern
  :: (NeuralNet n)
  => n
  -> LabelledImage
  -> Int
evalOnePattern net trainingData = isMatch result target
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

readTrainingData :: IO [LabelledImage]
readTrainingData = do
  putStrLn "Reading training labels..."
  trainingLabels <- readLabels' "data/trainForHaskell.csv"
  putStrLn $ "Read " ++ show (length trainingLabels) ++ " labels"
  putStrLn "Reading training images..."
  trainingImages <- readImages' "data/train" (map fst trainingLabels)
  putStrLn $ "Read " ++ show (length trainingImages) ++ " images"
  return $ zip (map normalisedData trainingImages) (map snd trainingLabels)

readTestData :: IO [LabelledImage]
readTestData = do
  putStrLn "Reading test labels..."
  testLabels <- readLabels "t10k-labels-idx1-ubyte"
  putStrLn $ show $ take 10 testLabels
--  putStrLn $ "Read " ++ show (length testLabels) ++ " labels"
--  putStrLn "Reading test images..."
  testImages <- readImages "t10k-images-idx3-ubyte"
--  putStrLn $ "Read " ++ show (length testImages) ++ " images"
--  putStrLn "Testing..."
  return (zip (map normalisedData testImages) testLabels)

