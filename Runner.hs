{-# OPTIONS_GHC -Wall                    #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing #-}
{-# OPTIONS_GHC -fno-warn-type-defaults  #-}

module Runner
  ( readLabels'
  , readTrainingData
  , trainWithAllPatterns
  , readTestData
  , evalAllPatterns
  ) where

import NeuralNet
import MarineExplore

import Data.List
import Data.Maybe

import Control.Applicative

import Debug.Trace


targets :: [[Double]]
targets =
    [
        [0.9, 0.1]
      , [0.1, 0.9]
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
evalOnePattern net trainingData = trace (show input ++ "\n" ++
                                         show target ++ "\n" ++
                                         show rawResult) $ isMatch result target
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
  -- trainingLabels <- take 2000 <$> readLabels' "data/trainForHaskell.csv"
  trainingLabels <- take 100 <$> readLabels' "data/HorV.csv"
  putStrLn $ "Read " ++ show (length trainingLabels) ++ " labels" ++ show (take 10 trainingLabels)
  putStrLn "Reading training images..."
  -- trainingImages <- readImages' "data/train" (map fst trainingLabels)
  trainingImages <- readImages' "data/check" (map fst trainingLabels)
  putStrLn $ "Read " ++ show (length trainingImages) ++ " images"
  return $ zip (map normalisedData trainingImages) (map snd trainingLabels)

readTestData :: IO [LabelledImage]
readTestData = do
  putStrLn "Reading test labels..."
  -- testLabels <- take 2000 <$> readLabels' "data/trainForHaskell.csv"
  testLabels <- take 100 <$> readLabels' "data/HorV.csv"
  putStrLn $ "Read " ++ show (length testLabels) ++ " labels"
  putStrLn "Reading test images..."
  -- testImages <- readImages' "data/train" (map fst testLabels)
  testImages <- readImages' "data/check" (map fst testLabels)
  putStrLn $ "Read " ++ show (length testImages) ++ " images"
  putStrLn "Testing..."
  return $ zip (map normalisedData testImages) (map snd testLabels)

