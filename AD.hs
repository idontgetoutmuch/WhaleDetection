{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveTraversable #-}

import Data.Array.Repa hiding (map)
import Numeric.AD
import Data.Foldable (Foldable)
import qualified Data.Foldable as Foldable
import Data.Traversable (Traversable)
import qualified Data.Traversable as Traversable

data MyMatrix a = MyMatrix
  {
    myRows :: Int
  , myCols :: Int
  , myElts :: [a]
  } deriving (Show, Functor, Foldable, Traversable)

f (MyMatrix r x es) = sum es

g :: MyMatrix Double -> Double
g (MyMatrix r c es) = head $ toList $ sumS $ sumS n
  where
    n   = fromListUnboxed (Z :. r :. c) es

