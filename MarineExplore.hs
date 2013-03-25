{-# OPTIONS_GHC -Wall                    #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing #-}
{-# OPTIONS_GHC -fno-warn-type-defaults  #-}

{-# LANGUAGE ScopedTypeVariables #-}

module MarineExplore
  ( Image(..)
  , normalisedData
  , readImages'
  , readLabels'
  , toMatrix
  )  where

import Data.Word
import qualified Data.List.Split as S
import Numeric.LinearAlgebra hiding ((<.>))
import System.FilePath
import System.IO

data Image = Image {
      iRows :: Int
    , iColumns :: Int
    , iPixels :: [Word8]
    } deriving (Eq, Show)

toMatrix :: Image -> Matrix Double
toMatrix image = (r><c) p :: Matrix Double
  where r = iRows image
        c = iColumns image
        p = map fromIntegral (iPixels image)

normalisedData :: Image -> [Double]
normalisedData image = map normalisePixel (iPixels image)

normalisePixel :: Word8 -> Double
normalisePixel p = (fromIntegral p) / 255.0

readLabels' :: FilePath -> IO [(Int, Int)]
readLabels' fileName = do
  csvData <- readFile fileName
  let ncs = map (\x  -> (x!!0, x!!1)) $
            map (S.splitOn ",") $ lines $ csvData
      nns :: [Int]
      nns =  map read $ map (drop 5) $ map head $ map (S.splitOn ".") $ map fst ncs
      cs  :: [Int]
      cs = map read $ map snd ncs
  return $ zip nns cs

readImages' :: FilePath -> [Int] -> IO [Image]
readImages' fileName is =
  mapM (\i -> readImage $ fileName </> ("MNIST" ++ show i) <.> "txt") is

readImage :: FilePath -> IO Image
readImage fileName =
  withFile fileName ReadMode
    (\h -> do content <- hGetContents h
              let rows = lines content
                  pixelss :: [[Word8]]
                  pixelss = map (\l -> map read $ S.splitOn " " l) rows
              -- FIXME: It seems if we don't force(?) evaluation
              -- with this putStrLn then we get strange errors -
              -- urk.
              putStrLn $ show (length pixelss) ++ ":" ++ show (length $ head pixelss)
              return $
                Image { iRows    = length rows
                      , iColumns = length $ head pixelss
                      , iPixels  = concat pixelss
                      }
    )