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

points = map p2 [ (0, 3), (3, 3.4), (6, 3), (5.75, 5.75), (9, 3.75), (12, 3)
                , (11.75, 5.75), (3, 0), (2,2), (6, 0.5), (9, 0), (12.25, 0.25)]

ds :: [Diagram B R2]
ds = [ (text "1" <> state)  # named "1"
       , label "0-9" 0.5
       , (text "2" <> state)  # named "2"
       , label "0-9" 0.5
       , label "." 1
       , (text "3" <> fState) # named "3"
       , label "0-9" 0.5
       , (text "4" <> state)  # named "4"
       , label "." 1
       , label "0-9" 0.5
       , (text "5" <> fState) # named "5"
       , label "0-9" 0.5]

label txt size = text txt # fontSize size

states = position (zip points ds)

shaft = arc 0 (1/6 :: Turn)
shaft' = arc 0 (1/2 :: Turn) # scaleX 0.33
line = trailFromOffsets [unitX]

arrowStyle1  = (with
                & arrowHead .~ noHead
                & tailSize .~ 0.03
                & arrowShaft .~ line
                & arrowTail .~ noTail
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

ninePointStencil :: Diagram B R2
ninePointStencil = points #
                   connWithLab "Centre" "West2" "West" (1/2 :: Turn) (0 :: Turn)   #
                   connWithLab "East" "East2" "Centre" (1/2 :: Turn) (0 :: Turn)   #
                   connWithLab "Centre" "SouthWest2" "SouthWest" (5/8 :: Turn) (1/8 :: Turn) #
                   connWithLab "Centre" "NorthWest2" "NorthWest" (3/8 :: Turn) (7/8 :: Turn)

  where
    connWithLab end1 centre end2 t1 t2 = connectStencil end1 centre t1 t2 .
                                         connectPerim' arrowStyle1 centre end2 t1 t2

    point l t n p = (text l # scale 0.03 <> t # named n) # translate p

    points =
             (point "x1" bndPt "NorthWest"  (r2 (0.0,  1.0)))  <>
             (point "w1" arrLb "NorthWest2" (r2 (0.25, 0.75))) <>
             (point "x2" bndPt "West"       (r2 (0.0,  0.5)))  <>
             (point "w2" arrLb "West2"      (r2 (0.25, 0.5)))  <>
             (point "x3" bndPt "SouthWest"  (r2 (0.0,  0.0)))  <>
             (point "w3" arrLb "SouthWest2" (r2 (0.25, 0.25))) <>
             (point "a"  intPt "Centre"     (r2 (0.5,  0.5)))  <>
             (point "y"  outPt "East"       (r2 (1.0,  0.5)))  <>
             (point "f"  arrLb "East2"      (r2 (0.75, 0.5)))

    connectStencil n1 n2 o1 o2 =
      connectPerim' arrowStyle3 n1 n2 o1 o2

    intPt = circle cSize # fc blue      # lw 0
    bndPt = circle cSize # fc red       # lw 0
    arrLb = square (2*cSize) # fc green # lw 0
    outPt = circle cSize # fc yellow    # lw 0

cSize :: Double
cSize = 0.03

main = mainWith ninePointStencil