module AM (
    foo
  , errDiag ) where

import System.Environment(getArgs)
import Graphics.Rendering.Chart
import Data.Colour
import Data.Colour.Names
import Data.Default.Class
import Data.Monoid
import Graphics.Rendering.Chart.Backend.Cairo hiding (runBackend, defaultEnv)
import Graphics.Rendering.Chart.Backend.Diagrams
import Diagrams.Core.Types hiding (render)
import Diagrams.TwoD.Types
import Diagrams.Backend.Cairo
import Control.Lens

import System.IO.Unsafe

setLinesBlue :: PlotLines a b -> PlotLines a b
setLinesBlue = plot_lines_style  . line_color .~ opaque blue

foo :: QDiagram Cairo R2 Any
foo = fst $ runBackend bar (render chart (500, 500))

bar :: DEnv
bar = unsafePerformIO $ defaultEnv vectorAlignmentFns 500 500

errDiag :: QDiagram Cairo R2 Any
errDiag = fst $ runBackend bar (render errChart (500, 500))

chart = toRenderable layout
  where
    am :: Double -> Double
    am x = (sin (x*3.14159/45) + 1) / 2 * (sin (x*3.14159/5))

    sinusoid1 = plot_lines_values .~ [[ (x,(am x)) | x <- [0,(0.5)..400]]]
              $ plot_lines_style  . line_color .~ opaque blue
              $ plot_lines_title .~ "am"
              $ def

    sinusoid2 = plot_points_style .~ filledCircles 2 (opaque red)
              $ plot_points_values .~ [ (x,(am x)) | x <- [0,7..400]]
              $ plot_points_title .~ "am points"
              $ def

    layout = layout1_title .~ "Amplitude Modulation"
           $ layout1_plots .~ [Left (toPlot sinusoid1),
                               Left (toPlot sinusoid2)]
           $ def

errChart = toRenderable layout
  where
    sinusoid1 = plot_lines_values .~ [[ (negate $ logBase 2 x,(negate $ logBase 2 $ numericalErr 1.0 x))
                                      | x <- take 40 powersOf2]]
              $ plot_lines_style  . line_color .~ opaque blue
              $ plot_lines_title .~ "error"
              $ def

    layout = layout1_title .~ "Floating Point Error"
           $ layout1_plots .~ [Left (toPlot sinusoid1)]
           $ layout1_left_axis .~ errorAxis
           $ layout1_bottom_axis .~ stepSizeAxis
           $ def

    errorAxis = laxis_title .~ "Minus log to base 2 of the error"
              $ def

    stepSizeAxis = laxis_title .~ "Minus log to base 2 of the step size"
                 $ def


f :: Double -> Double
f x = exp x

numericalF' :: Double -> Double -> Double
numericalF' x h = (f (x + h) - f x) / h

numericalErr :: Double -> Double -> Double
numericalErr x h = abs ((exp 1.0 - numericalF' x h))

powersOf2 :: Fractional a => [a]
powersOf2 = 1 : map (/2) powersOf2

errs = map (logBase 2 . numericalErr 1.0) powersOf2

main1 ["small"]  = renderableToPNGFile chart 320 240 "example1_small.png"
main1 ["big"]    = renderableToPNGFile chart 800 600 "example1_big.png"

main = getArgs >>= main1
