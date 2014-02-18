{-# LANGUAGE NoMonomorphismRestriction #-}

import Data.Maybe (fromMaybe)
import Diagrams.Prelude
import Diagrams.TwoD.Arrow
import Diagrams.Backend.CmdLine
-- import Diagrams.Backend.SVG.CmdLine
import Diagrams.Backend.Cairo.CmdLine

state :: Diagram B R2
state = circle 1 # lw 0.05 # fc silver

fState :: Diagram B R2
fState = circle 0.85 # lw 0.05 # fc lightblue <> state

label txt size = text txt # fontSize size

t  = cubicSpline False (map p2 [(0, 0), (1, 0), (1, 0.2), (2, 0.2)])
line = trailFromOffsets [unitX]

arrowStyle1  = (with
                & arrowHead .~ noHead
                & tailSize .~ 0.03
                & arrowShaft .~ line
                & arrowTail .~ noTail
                & tailColor  .~ black)

arrowStyle2  = (with
                & arrowHead .~ noHead
                & tailSize .~ 0.03
                & arrowShaft .~ t
                & arrowTail .~ spike'
                & tailColor  .~ black)

arrowStyle3  = (with
                & arrowHead .~ noHead
                & tailSize .~ 0.03
                & arrowShaft .~ line
                & arrowTail .~ spike'
                & tailColor  .~ black)

displayHeader :: FilePath -> Diagram B R2 -> IO ()
displayHeader fn =
  mainRender ( DiagramOpts (Just 900) (Just 600) fn
             , DiagramLoopOpts False Nothing 0
             )

perceptron :: Int -> Diagram B R2
perceptron pN = points #
                connWithLab "Centre" "West2" "West" (1/2 :: Turn) (0 :: Turn)   #
                connWithLab "East" "East2" "Centre" (1/2 :: Turn) (0 :: Turn)   #
                connWithLab "Centre" "SouthWest2" "SouthWest" (5/8 :: Turn) (1/8 :: Turn) #
                connWithLab "Centre" "NorthWest2" "NorthWest" (3/8 :: Turn) (7/8 :: Turn)

  where
    connWithLab end1 centre end2 t1 t2 =
      connectStencil (end1, pN) (centre, pN) t1 t2 .
      connectPerim' arrowStyle1 (centre, pN) (end2, pN) t1 t2

    point l t n p = (text l # scale 0.03 <> t # named (n, pN)) # translate p

    points =
      (point "x1" bndPt "NorthWest" nwCoords)  <>
      (point "w1" arrLb "NorthWest2" (0.5 * (nwCoords + cCoords))) <>
      (point "x2" bndPt "West"       (r2 (0.0,  0.5)))  <>
      (point "w2" arrLb "West2"      (r2 (0.25, 0.5)))  <>
      (point "x3" bndPt "SouthWest"  (r2 (0.0,  0.0)))  <>
      (point "w3" arrLb "SouthWest2" (r2 (0.25, 0.25))) <>
      (point "a"  intPt "Centre"     cCoords)  <>
      (point "y"  outPt "East"       (r2 (1.0,  0.5)))  <>
      (point "f"  arrLb "East2"      (r2 (0.75, 0.5)))

    nwCoords = r2 (0.0,  1.0)
    cCoords =  r2 (0.5,  0.5)

    connectStencil n1 n2 o1 o2 =
      connectPerim' arrowStyle3 n1 n2 o1 o2

    intPt = circle cSize # fc blue      # lw 0
    bndPt = circle cSize # fc red       # lw 0
    arrLb = square (2*cSize) # fc green # lw 0
    outPt = circle cSize # fc yellow    # lw 0

cSize :: Double
cSize = 0.03

multiPerceptron =
  ((perceptron 1) <>
   ((perceptron 2) # translate (r2 (1.5, 1.5))) <>
   ((perceptron 3) # translate (r2 (0.0, 1.5))) <>
   ((perceptron 4) # translate (r2 (0.0, 3.0)))
  ) #
  connectPerim' arrowStyle2 ("SouthWest", 2 :: Int) ("East", 1 :: Int) (3/4 :: Turn) (1/4 :: Turn) #
  connectPerim' arrowStyle2 ("West", 2 :: Int) ("East", 3 :: Int) (1/2 :: Turn) (0 :: Turn) #
  connectPerim' arrowStyle2 ("NorthWest", 2 :: Int) ("East", 4 :: Int) (1/4 :: Turn) (3/4 :: Turn)
main = do
  displayHeader "/Users/dom/Dropbox/Private/Whales/diagrams/Fita1.png"
    (perceptron 1)
  displayHeader "/Users/dom/Dropbox/Private/Whales/diagrams/Fita2.png"
    multiPerceptron
