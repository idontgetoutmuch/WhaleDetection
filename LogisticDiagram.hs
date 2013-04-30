{-# LANGUAGE TupleSections #-}

import Diagrams.Prelude

import Diagrams.Backend.Cairo.CmdLine
import Data.Colour (withOpacity)

import Data.Random ()
import Data.Random.Distribution.Beta
import Data.RVar

import System.Random

import Data.List
import qualified Data.IntMap as IntMap

import Control.Monad.State

import Text.Printf

tickSize   = 0.1
nCells     = 100
a          = 15
b          = 6
nSamples   = 100000
cellColour0 = red  `withOpacity` 0.5
cellColour1 = blue `withOpacity` 0.5

background = rect 1.1 1.1 # translate (r2 (0.5, 0.5))

ticks xs = (mconcat $ map tick xs)  <> line
  where
    maxX   = maximum xs
    line   = fromOffsets [r2 (maxX, 0)]
    tSize  = maxX / 100
    tick x = endpt # translate tickShift
      where
        tickShift = r2 (x, 0)
        endpt     = topLeftText (printf "%.2f" x) # fontSize (tSize * 2) <>
                    circle tSize # fc red  # lw 0

betas :: Int -> Double -> Double -> [Double]
betas n a b =
  fst $ runState (replicateM n (sampleRVar (beta a b))) (mkStdGen seed)
    where
      seed = 0

histogram :: Int -> [Double] -> IntMap.IntMap Int
histogram nCells xs =
  foldl' g emptyHistogram xs
    where
      g h x          = IntMap.insertWith (+) (makeCell nCells x) 1 h
      emptyHistogram = IntMap.fromList $ zip [0 .. nCells - 1] (repeat 0)
      makeCell m     = floor . (* (fromIntegral m))

hist cellColour xs = scaleX sX . scaleY sY . position $ hist' where
    ysmax = fromInteger . ceiling $ maximum xs
    ysmin = fromInteger . floor $ minimum xs
    xsmax = fromIntegral $ length xs
    xsmin = 0.0
    sX = 1 / (xsmax - xsmin)
    sY = 1 / (ysmax - ysmin)
    hist' = zip (map p2 $ map (,0) $
            map fromInteger [0..]) (map (cell 1) xs)
    cell w h = alignB $ rect w h
                      # fcA cellColour
                      # lc white
                      # lw 0.001

test tickSize nCells a b nSamples =
  ticks [0.0, tickSize..1.0] <>
  hist cellColour0 ys <>
  hist cellColour1 xs <>
  background
    where
      xs = IntMap.elems $
           IntMap.map fromIntegral $
           histogram nCells $
           betas nSamples a b
      ys = IntMap.elems $
           IntMap.map fromIntegral $
           histogram nCells $
           betas nSamples b a

main :: IO ()
main = defaultMain $ test tickSize nCells a b nSamples
