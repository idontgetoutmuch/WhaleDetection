module MLTalkDiagrams (
    example
  , foo
  , errDiag ) where

import Data.List.Split
import Data.Maybe
import Diagrams.Backend.Cairo
import Diagrams.Backend.Cairo.CmdLine
import Diagrams.BoundingBox
import Diagrams.Core.Envelope
import Diagrams.Coordinates
import Diagrams.Prelude
import Graphics.SVGFonts

import AM ( foo, errDiag )

example = let c = cube
          in pad 1.1 . centerXY $ c <> drawLines c

box innards padding colour =
    let padded =                  strutY padding
                                       ===
             (strutX padding ||| centerXY innards ||| strutX padding)
                                       ===
                                  strutY padding
        height = diameter (r2 (0,1)) padded
        width  = diameter (r2 (1,0)) padded
    in centerXY innards <> roundedRect width height 0.1 # fcA (colour `withOpacity` 0.1)

textOpts s n = TextOpts s lin2 INSIDE_H KERN False 1 n

text' :: String -> Double -> Diagram Cairo R2
text' s n = textSVG_ (textOpts s n) # fc black # lw 0

centredText ls n = vcat' with { catMethod = Distrib, sep = (n) }
                     (map (\l -> centerX (text' l n)) ls)
centredText' s = centredText (splitOn "\n" s)

padAmount = 0.5
padAmount' = 1.0

down :: R2
down = (0& (-3))

upright :: R2
upright = (7&5)

right :: R2
right = (5&0)

left :: R2
left = ((-5)&0)

fun s n = (box (centredText' s 1) padAmount blue) # named n
var s n = (box (centredText' s 1) padAmount red) # named n

cube :: Diagram Cairo R2
cube = mconcat
  [ var "u1" "u1name"      # translate right
  , fun "exp (_)" "exp1" # translate (right ^+^ down)
  , var "u2" "u2name"      # translate (right ^+^ down ^+^
                                        down)
  , fun "(_)^2" "^2"     # translate (right ^+^ down ^+^
                                      down ^+^ right)
  , fun "+" "+1"         # translate (right ^+^ down ^+^
                                      down ^+^ down)
  , var "u3" "u3name"      # translate (right ^+^ down ^+^
                                        down ^+^ right ^+^
                                        down)
  , var "u4" "u4name"      # translate (right ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down)
  , fun "exp (_)" "exp2" # translate (right ^+^ down ^+^
                                      down ^+^ down ^+^
                                      down ^+^ down ^+^
                                      left)
  , fun "sin (_)" "sin"  # translate (right ^+^ down ^+^
                                      down ^+^ down ^+^
                                      down ^+^ down ^+^
                                      right)
  , var "u6" "u6name"      # translate (right ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down ^+^ left)
  , var "u5" "u5name"      # translate (right ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down ^+^ right)
  , fun "+" "+2"         # translate (right ^+^ down ^+^
                                      down ^+^ down ^+^
                                      down ^+^ down ^+^
                                      down ^+^ down)
  , var "u7" "u7name"      # translate (right ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down ^+^ down ^+^
                                        down)
               ]

drawLines :: Diagram Cairo R2 -> Diagram Cairo R2
drawLines cube = foldr (.) id (map (uncurry connect) pairs) cube
  where pairs = [ ("u1name", "exp1")
                , ("exp1",   "u2name")
                , ("u2name", "+1")
                , ("u2name", "^2")
                , ("^2",     "u3name")
                , ("u3name", "+1")
                , ("+1",     "u4name")
                , ("u4name", "exp2")
                , ("u4name", "sin")
                , ("exp2",   "u6name")
                , ("sin",    "u5name")
                , ("u6name", "+2")
                , ("u5name", "+2")
                , ("+2",     "u7name")
                ]

connect n1 n2 = withName n1 $ \b1 ->
                withName n2 $ \b2 ->
                  let v = location b2 .-. location b1
                      midpoint = location b1 .+^ (v/2)
                      p1 = fromJust $ traceP midpoint (-v) b1
                      p2 = fromJust $ traceP midpoint v b2
                  in atop (arrow p1 p2)

arrow p1 p2 = (p1 ~~ p2) <> arrowhead
  where v = p2 .-. p1
        arrowhead = eqTriangle 0.5
                    # alignT
                    # fc black
                    # rotateBy (direction v - 1/4)
                    # moveTo p2