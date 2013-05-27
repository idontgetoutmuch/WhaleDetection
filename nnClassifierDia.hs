{-# LANGUAGE NoMonomorphismRestriction #-}

module NnClassifierDia (
  nn
  ) where

import Diagrams.Prelude

nn = mconcat layer1s <>
     mconcat layer2s <>
     mconcat arrow1s <>
     mconcat arrow2s <>
     mconcat layer3s <>
     background
       where
         layer1s = zipWith (myCircle' 0.1) [0.1,0.3..0.9] ["x1", "x2", "x3", "x4", "x5"]
         arrow1s = map (\x -> drawV 0.1 x (0.8 *^ getDirection (0.1, x) (0.5, 0.5))) [0.1,0.3..0.9]
         layer2s = zipWith (myCircle'' 0.5) [0.5] ["a"]
         arrow2s = [drawV 0.5 0.5 (0.8 *^ getDirection (0.5, 0.5) (0.8, 0.5))]
         layer3s = [text "y = f(a)" # scale 0.05 # translate (r2 (0.9, 0.5))]

getDirection (x1, y1) (x2, y2) =
  (x2 - x1) & (y2 - y1)

drawV x y v = (arrowHead <> shaft) # fc black # translate (r2 (x, y))
   where
    shaft     = origin ~~ (origin .+^ v)
    arrowHead = eqTriangle 0.01
              # rotateBy (direction v - 1/4)
              # translate v

myCircle' x y l =
  (t # translate (r2 (x - 0.1, y + 0.1))) <>
  (circle 0.05 # fc blue # translate (r2 (x, y)) # opacity 0.5)
  where
    t = text l # scale 0.05

myCircle'' x y l =
  (t # translate (r2 (x, y + 0.1))) <>
  (circle 0.05 # fc blue # translate (r2 (x, y)) # opacity 0.5)
  where
    t = text l # scale 0.05

background = rect 1.2 1.2 # translate (r2 (0.5, 0.5)) # fc beige # opacity 0.5
