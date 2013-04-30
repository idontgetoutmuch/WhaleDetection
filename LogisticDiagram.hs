One way of visualizing a distribution is to take lots of sample for it
and plot the resulting histogram.

To do this we use the
[diagrams](http://hackage.haskell.org/package/diagrams) package.

~~~~ { .haskell }
{-# LANGUAGE TupleSections #-}

import Diagrams.Prelude

import Diagrams.Backend.Cairo.CmdLine
import Data.Colour (withOpacity)
~~~~

To generate the actual samples (in this case from the [beta
distribution](http://en.wikipedia.org/wiki/Beta_distribution)) we use
the [random-fu](http://hackage.haskell.org/package/random-fu-0.2.3.1)
and [rvar](http://hackage.haskell.org/package/rvar-0.2.0.1) packages.

~~~~ { .haskell }
import Data.Random ()
import Data.Random.Distribution.Beta
import Data.RVar
~~~~

We import some other standard modules and define some constants.

~~~~ { .haskell }
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
cellColour = blue `withOpacity` 0.5
~~~~

First we define a background on which we are going to plot our
histogram and also a function to draw the x-axis.

~~~~ { .haskell }
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
~~~~

We sample from the beta distribution with parameters $a$ and $b$ and
put these into a histogram.

~~~~ { .haskell }
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
~~~~

Finally we can render the histogram by drawing blue rectangles with a
white border.

~~~~ { .haskell }
hist xs = scaleX sX . scaleY sY . position $ hist' where
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
~~~~

And now we can draw our histogram.

~~~~ { .haskell }
test tickSize nCells a b nSamples =
  ticks [0.0, tickSize..1.0] <>
  hist xs <>
  background
    where
      xs = IntMap.elems $
           IntMap.map fromIntegral $
           histogram nCells $
           betas nSamples a b

main :: IO ()
main = defaultMain $ test tickSize nCells a b nSamples
~~~~

```{.dia width='800'}
{-# LANGUAGE TupleSections #-}

import Diagrams.Prelude

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
cellColour = blue `withOpacity` 0.5

background = rect 1.1 1.1 # translate (r2 (0.5, 0.5))

test tickSize nCells a b nSamples =
  ticks [0.0, tickSize..1.0] <>
  hist xs <>
  background
    where
      xs = IntMap.elems $
      	   IntMap.map fromIntegral $
           histogram nCells $
           betas nSamples a b

hist xs = scaleX sX . scaleY sY . position $ hist' where
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

dia = test tickSize nCells a b nSamples
```
