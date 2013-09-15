module AM (foo) where

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

main1 ["small"]  = renderableToPNGFile chart 320 240 "example1_small.png"
main1 ["big"]    = renderableToPNGFile chart 800 600 "example1_big.png"

main = getArgs >>= main1
