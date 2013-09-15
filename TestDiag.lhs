> module TestDiag (
>     main
>   , example
>   , foo ) where

> import Data.List.Split
> import Data.Maybe
> import Diagrams.Backend.Cairo
> import Diagrams.Backend.Cairo.CmdLine
> import Diagrams.BoundingBox
> import Diagrams.Core.Envelope
> import Diagrams.Coordinates
> import Diagrams.Prelude
> import Graphics.SVGFonts
>
> import AM (foo)

```{.dia width='500'}
import TestDiag
dia = foo
```

```{.dia width='500'}
import TestDiag
dia = example
```

The diagram is the boxes (the "cube") and the lines between the boxes.

> example = let c = cube
>           in pad 1.1 . centerXY $ c <> drawLines c

A "box" is a diagram (the "innards") surrounded by a rounded
rectangle.  First the innards are padded by a fixed amount, then we
compute its height and width -- that's the size of the surrounding
rectangle.

> box innards padding colour =
>     let padded =                  strutY padding
>                                        ===
>              (strutX padding ||| centerXY innards ||| strutX padding)
>                                        ===
>                                   strutY padding
>         height = diameter (r2 (0,1)) padded
>         width  = diameter (r2 (1,0)) padded
>     in centerXY innards <> roundedRect width height 0.1 # fcA (colour `withOpacity` 0.1)
>
> textOpts s n = TextOpts s lin2 INSIDE_H KERN False 1 n

A single string of text.

> text' :: String -> Double -> Diagram Cairo R2
> text' s n = textSVG_ (textOpts s n) # fc black # lw 0

Several lines of text stacked vertically.

> centredText ls n = vcat' with { catMethod = Distrib, sep = (n) }
>                      (map (\l -> centerX (text' l n)) ls)
> centredText' s = centredText (splitOn "\n" s)

Diagram-specific parameters, including the positioning vectors.

> padAmount = 0.5
> padAmount' = 1.0
>
> down :: R2
> down = (0& (-3))
>
> upright :: R2
> upright = (7&5)
>
> right :: R2
> right = (5&0)
>
> left :: R2
> left = ((-5)&0)

A box with some interior text and a name.

> fun s n = (box (centredText' s 1) padAmount blue) # named n
> var s n = (box (centredText' s 1) padAmount red) # named n

The cube is just several boxes superimposed, positioned by adding
together some positioning vectors.

> cube :: Diagram Cairo R2
> cube = mconcat
>   [ var "z" "zname"      # translate right
>   , fun "exp (_)" "exp1" # translate (right ^+^ down)
>   , var "y" "yname"      # translate (right ^+^ down ^+^
>                                       down)
>   , fun "(_)^2" "^2"     # translate (right ^+^ down ^+^
>                                       down ^+^ right)
>   , fun "+" "+1"         # translate (right ^+^ down ^+^
>                                       down ^+^ down)
>   , var "x" "xname"      # translate (right ^+^ down ^+^
>                                       down ^+^ right ^+^
>                                       down)
>   , var "w" "wname"      # translate (right ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down)
>   , fun "exp (_)" "exp2" # translate (right ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       left)
>   , fun "sin (_)" "sin"  # translate (right ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       right)
>   , var "u" "uname"      # translate (right ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ left)
>   , var "v" "vname"      # translate (right ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ right)
>   , fun "+" "+2"         # translate (right ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down)
>   , var "f" "fname"      # translate (right ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down ^+^ down ^+^
>                                       down)
>                ]

For each pair (a,b) of names, draw an arrow from diagram "a" to
diagram "b".

> drawLines :: Diagram Cairo R2 -> Diagram Cairo R2
> drawLines cube = foldr (.) id (map (uncurry connect) pairs) cube
>   where pairs = [ ("zname","exp1")
>                 , ("exp1", "yname")
>                 , ("yname", "+1")
>                 , ("yname", "^2")
>                 , ("^2", "xname")
>                 , ("xname", "+1")
>                 , ("+1", "wname")
>                 , ("wname", "exp2")
>                 , ("wname", "sin")
>                 , ("exp2", "uname")
>                 , ("sin", "vname")
>                 , ("uname", "+2")
>                 , ("vname", "+2")
>                 , ("+2", "fname")
>                 ]

Draw an arrow from diagram named "n1" to diagram named "n2".  The
arrow lies on the line between the centres of the diagrams, but is
drawn so that it stops at the boundaries of the diagrams, using traces
to find the intersection points.

> connect n1 n2 = withName n1 $ \b1 ->
>                 withName n2 $ \b2 ->
>                   let v = location b2 .-. location b1
>                       midpoint = location b1 .+^ (v/2)
>                       p1 = fromJust $ traceP midpoint (-v) b1
>                       p2 = fromJust $ traceP midpoint v b2
>                   in atop (arrow p1 p2)

Draw an arrow from point p1 to point p2.  An arrow is just a line with
a triangle at the head.

> arrow p1 p2 = (p1 ~~ p2) <> arrowhead
>   where v = p2 .-. p1
>         arrowhead = eqTriangle 0.5
>                     # alignT
>                     # fc black
>                     # rotateBy (direction v - 1/4)
>                     # moveTo p2

> main :: IO ()
> main = defaultMain example