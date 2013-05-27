{-# LANGUAGE NoMonomorphismRestriction #-}

module NnClassifierDia (
  nn
  ) where

import Diagrams.Prelude

maxLayer = 5
intraLayerGap = 1.0 / (fromIntegral maxLayer)

nodesInt :: Int -> [Int]
nodesInt 0 = [0]
nodesInt n = [negate n] ++ nodesInt (n - 1) ++ [n]

nodes n intraLayerGap = map (+0.5) $ map (*intraLayerGap) $ map fromIntegral $ nodesInt n

nn = mconcat (layer1s 3) <>
     mconcat (mconcat $ map (arrow1s 3) $ nodes 1 intraLayerGap) <>
     mconcat layer2s <>
     mconcat (mconcat $ map arrow2s $ nodes 2 intraLayerGap) <>
     mconcat layer3s <>
     background
       where
         layer1s n = zipWith (myCircle' xOff) ns ms
           where
             ns = nodes n (1 / (2 * (fromIntegral n) + 1))
             ms = zipWith (\x y -> 'x':(show x)) [1..] ns
             xOff = head ns
         arrow1s n y = map (\x -> drawV 0.1 x (0.8 *^ getDirection (0.1, x) (0.5, y))) ns
           where
             ns = nodes n (1 / (2 * (fromIntegral n) + 1))
         arrow2s y = map (\x -> drawV 0.5 x (0.8 *^ getDirection (0.5, x) (0.8, y))) (nodes 1 intraLayerGap)
         layer2s = zipWith (myCircle'' 0.5) (nodes 1 intraLayerGap) ["z1,1", "z1,2", "z1,3"]
         layer3s = zipWith (myCircle'' 0.85) (nodes 2 intraLayerGap) ["z2,1", "z2,2", "z2,3", "z2,4", "z2,5"]

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
