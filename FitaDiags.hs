{-# LANGUAGE NoMonomorphismRestriction #-}

import Data.Maybe (fromMaybe)
import Diagrams.Prelude
import Diagrams.TwoD.Arrow
import Diagrams.Backend.CmdLine
import Diagrams.Backend.SVG.CmdLine

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

arrowStyle1 = (with  & arrowHead  .~ noHead & tailSize .~ 0.3
                     & arrowShaft .~ shaft  & arrowTail .~ spike'
                     & tailColor  .~ black)

arrowStyle2  = (with  & arrowHead  .~ noHead &  tailSize .~ 0.3
                      & arrowShaft .~ shaft' & arrowTail .~ spike'
                      & tailColor  .~ black)

arrowStyle3  = (with  & arrowHead  .~ noHead & tailSize .~ 0.3
                      & arrowShaft .~ line & arrowTail .~ spike'
                      & tailColor  .~ black)

example = states # connectPerim' arrowStyle1
                                 "2" "1" (5/12 :: Turn) (1/12 :: Turn)
                 # connectPerim' arrowStyle3
                                 "4" "1" (2/6 :: Turn) (5/6 :: Turn)
                 # connectPerim' arrowStyle2
                                 "2" "2" (2/12 :: Turn) (4/12 :: Turn)
                 # connectPerim' arrowStyle1
                                 "3" "2" (5/12 :: Turn) (1/12 :: Turn)
                 # connectPerim' arrowStyle2
                                 "3" "3" (2/12 :: Turn) (4/12 :: Turn)
                 # connectPerim' arrowStyle1
                                 "5" "4" (5/12 :: Turn) (1/12 :: Turn)
                 # connectPerim' arrowStyle2
                                 "5" "5" (-1/12 :: Turn) (1/12 :: Turn)

displayHeader :: FilePath -> Diagram B R2 -> IO ()
displayHeader fn =
  mainRender ( DiagramOpts (Just 900) (Just 600) fn
             , DiagramLoopOpts False Nothing 0
             )


arrowStyle4 :: Color a => a -> ArrowOpts
arrowStyle4 c = (with  & arrowHead  .~ noHead
                       & shaftStyle %~ lw 0.005
                       & shaftColor .~ c
                       & arrowTail  .~ noTail)

ninePointStencil :: Diagram B R2
ninePointStencil = points #
                   connectStencil "Centre" "West"  (1/2 :: Turn) (0 :: Turn)   #
                   connectStencil "East" "Centre"  (1/2 :: Turn) (0 :: Turn)   #
                   connectStencil "Centre" "North" (1/4 :: Turn) (3/4 :: Turn) #
                   connectStencil "South" "Centre" (1/4 :: Turn) (3/4 :: Turn) #
                   connectStencil "SouthWest" "Centre" (1/8 :: Turn) (5/8 :: Turn) #
                   connectStencil "SouthEast" "Centre" (3/8 :: Turn) (7/8 :: Turn) #
                   connectStencil "NorthEast" "Centre" (5/8 :: Turn) (1/8 :: Turn) #
                   connectStencil "NorthWest" "Centre" (7/8 :: Turn) (3/8 :: Turn)

  where points = ((text "W" # scale 0.03 <> bndPt # named "West") # translate (r2 (0.0, 0.5))) <>
                 (intPt # named "Centre" # translate (r2 (0.5, 0.5))) <>
                 ((text "E" # scale 0.03 <> bndPt # named "East") # translate (r2 (1.0, 0.5))) <>
                 (bndPt # named "North" # translate (r2 (0.5, 1.0))) <>
                 (bndPt # named "South" # translate (r2 (0.5, 0.0))) <>
                 (bndPt # named "SouthWest" # translate (r2 (0.0, 0.0))) <>
                 (bndPt # named "NorthWest" # translate (r2 (0.0, 1.0))) <>
                 (bndPt # named "SouthEast" # translate (r2 (1.0, 0.0))) <>
                 (bndPt # named "NorthEast" # translate (r2 (1.0, 1.0)))

        connectStencil n1 n2 o1 o2 =
          connectPerim' arrowStyle3 {- (arrowStyle4 green) -} n1 n2 o1 o2

        intPt = circle cSize # fc blue # lw 0
        bndPt = circle cSize # fc red  # lw 0

cSize :: Double
cSize = 0.03

main = mainWith example