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

nodes' n = nodes n intraLayerGap
  where
    intraLayerGap = 1 / (2 * (fromIntegral n) + 1)

nn = mconcat (layer1s 3) <>
     mconcat (mconcat $ map (arrow1s 3) $ nodes' 1) <>
     mconcat (layer2s 1) <>
     mconcat (mconcat $ map (arrow2s 1) $ nodes' 2) <>
     mconcat layer3s <>
     mconcat (mconcat $ map (arrow3s 2) $ nodes' 1) <>
     mconcat layer4s <>
     background
       where
         layer1s n = zipWith (myCircle' xOff) ns ms
           where
             ns = nodes' n
             ms = zipWith (\x y -> 'x':(show x)) [1..] ns
             xOff = head ns

         arrow1s n y = map (\x -> drawV (cOff / 2) x (0.85 *^ getDirection (0.1, x) (0.5, y))) ns
           where
             cOff = 1 / (2 * (fromIntegral n) + 1)
             ns   = nodes' n

         layer2s n = zipWith (myCircle'' 0.5) ns ms
           where
             ns = nodes' n
             ms = zipWith (\x y -> 'z':'1':',':(show x)) [1..] ns

         arrow2s n y = map (\x -> drawV 0.5 x (0.9 *^ getDirection (0.5, x) (0.8, y))) (nodes' n)
         layer3s = zipWith (myCircle'' 0.85) ns ms
           where
             ns = nodes 2 intraLayerGap
             ms = zipWith (\x y -> 'z':'2':',':(show x)) [1..] ns

         arrow3s n y = map (\x -> drawV 0.85 x (0.9 *^ getDirection (0.5, x) (0.8, y))) (nodes' n)
         layer4s = zipWith (myCircle'' 1.2) ns ms
           where
             ns = nodes' 1
             ms = zipWith (\x y -> 'y':(show x)) [1..] ns

getDirection (x1, y1) (x2, y2) =
  (x2 - x1) & (y2 - y1)

drawV x y v = (arrowHead <> shaft) # fc black # translate (r2 (x, y))
   where
    shaft     = origin ~~ (origin .+^ v)
    arrowHead = eqTriangle 0.01
              # rotateBy (direction v - 1/4)
              # translate v

myCircle' x y l =
  (t # translate (r2 (x - 0.1, y + 0.05))) <>
  (circle 0.05 # fc blue # translate (r2 (x, y)) # opacity 0.5)
  where
    t = text l # scale 0.05

myCircle'' x y l =
  (t # translate (r2 (x, y + 0.1))) <>
  (circle 0.05 # fc blue # translate (r2 (x, y)) # opacity 0.5)
  where
    t = text l # scale 0.05

background = rect 1.6 1.2 # translate (r2 (0.6, 0.5)) # fc beige # opacity 0.5
