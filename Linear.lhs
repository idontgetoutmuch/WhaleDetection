% Regression and Automated Differentiation
% Dominic Steinitz
% 24th April 2013

Introduction
------------

Automated differentiation was developed in the 1960's but even now
does not seem to be that widely used. Even experienced and
knowledgeable practitioners often assume it is either a finite
difference method or symbolic computation when it is neither.

This article gives a very simple application of it in a machine
learning / statistics context.

Multivariate Linear Regression
------------------------------

We model a dependent variable linearly dependent on some set
of independent variables in a noisy environment.

$$
y^{(i)} = \boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)} + \epsilon^{(i)}
$$

where

* $i$ runs from 1 to $n$, the number of observations;

* $\epsilon^{(i)}$ are i.i.d. normal with mean $0$ and the same
variance $\sigma^2$: $\epsilon^{(i)} \sim \mathcal{N} (0,\sigma^2)$;

* For each $i$, $\boldsymbol{x^{(i)}}$ is a column vector of size $m$ and

* $\boldsymbol{\theta}$ is a column vector also of size $m$.

In other words:

$$
p(y^{(i)} \mid \boldsymbol{x}^{(i)}; \boldsymbol{\theta}) =
\frac{1}{\sqrt{2\pi}\sigma}\exp\big(\frac{-(y^{(i)} - \boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)})^2}{2\sigma^2}\big)
$$

We can therefore write the likelihood function given all the observations as:

$$
\mathcal{L}(\boldsymbol{\theta}; X, \boldsymbol{y}) =
\prod_{i = 1}^n \frac{1}{\sqrt{2\pi}\sigma}\exp\big(\frac{-(y^{(i)} - \boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)})^2}{2\sigma^2}\big)
$$

In order to find the best fitting parameters $\boldsymbol{\theta}$ we
therefore need to maximize this function with respect to
$\boldsymbol{\theta}$. The standard approach is to maximize the log
likelihood which, since log is monotonic, will give the same result.

$$
\begin{align*}
\mathcal{l}(\boldsymbol{\theta}) &= \log \mathcal{L}(\boldsymbol{\theta}) \\
                                 &= \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi}\sigma}\exp\big(\frac{-(y^{(i)} - \boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)})^2}{2\sigma^2}\big) \\
                                 &= n\log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}\sum_{i=1}^n (y^{(i)} - \boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)})^2
\end{align*}
$$

Hence maximizing the likelihood is the same as minimizing the (biased)
estimate of the variance:

$$
\frac{1}{n}\sum_{i=1}^n (y^{(i)} - \boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)})^2
$$

Some pragmas to warn us about potentially dangerous situations.

> {-# OPTIONS_GHC -Wall                    #-}
> {-# OPTIONS_GHC -fno-warn-name-shadowing #-}
> {-# OPTIONS_GHC -fno-warn-type-defaults  #-}

> module Main (main) where

Modules from the automatic differentiation [library][Package:ad].

  [Package:ad]: http://hackage.haskell.org/package/ad-3.4

> import Numeric.AD
> import Numeric.AD.Types

> import qualified Data.Vector as V

Some modules from a random number generator [library][Package:random-fu] as we will want to
generate some test data.

  [Package:random-fu]: http://hackage.haskell.org/package/random-fu-0.2.4.0

> import Data.Random ()
> import Data.Random.Distribution.Normal
> import Data.Random.Distribution.Uniform
> import Data.RVar

> yhat :: Floating a =>
>         V.Vector a ->
>         V.Vector a -> a
> yhat x theta = V.sum $ V.zipWith (*) theta x
>
> cost :: Floating a =>
>         V.Vector a ->
>         a ->
>         V.Vector a
>         -> a
> cost theta y x = 0.5 * (y - yhat x theta)^2
>
> delCost :: Floating a =>
>            a ->
>            V.Vector a ->
>            V.Vector a ->
>            V.Vector a
> delCost y x = grad $ \theta -> cost theta (auto y) (V.map auto x)
>
> totalCost :: Floating a =>
>              V.Vector a ->
>              V.Vector a ->
>              V.Vector (V.Vector a)
>              -> a
> totalCost theta y x = (/l) $ V.sum $ V.zipWith (cost theta) y x
>   where
>     l = fromIntegral $ V.length y

> delTotalCost :: Floating a =>
>                 V.Vector a ->
>                 V.Vector (V.Vector a) ->
>                 V.Vector a ->
>                 V.Vector a
> delTotalCost y x = grad $ \theta -> totalCost theta (V.map auto y) (V.map (V.map auto) x)

Although we only have two independent variables, we need three
parameters for the model.

> initTheta :: V.Vector Double
> initTheta = V.replicate 3 0.1

> theta0, theta1, theta2 :: Double
> theta0 = 0.0
> theta1 = 0.6
> theta2 = 0.7
>
> createSample :: IO [Double]
> createSample = do
>   x1 <- sampleRVar stdUniform
>   x2 <- sampleRVar stdUniform
>   let mu = theta0 + theta1 * x1 + theta2 * x2
>   y <- sampleRVar $ normal mu 0.01
>   return [y, x1, x2]
>
> nSamples, nIters:: Int
> nSamples = 100
> nIters = 2000

> gamma :: Double
> gamma = 0.1

> stepOnce :: V.Vector Double ->
>             V.Vector (V.Vector Double) ->
>             V.Vector Double ->
>             V.Vector Double
> stepOnce y x theta = V.zipWith (-) theta (V.map (* gamma) $ del theta)
>   where
>     del = delTotalCost y x

> stepOnceStoch :: Double ->
>                  V.Vector Double ->
>                  V.Vector Double ->
>                  V.Vector Double
> stepOnceStoch y x theta = V.zipWith (-) theta (V.map (* gamma) $ del theta)
>   where
>     del = delCost y x
>

> main :: IO ()
> main = do
>   vals <- sequence $ take nSamples $ repeat createSample
>   let y  = V.fromList $ map (!!0) vals
>       x  = V.fromList $ map (V.fromList) $ map (drop 1) vals
>       x' = V.map (V.cons 1.0) x
>       hs = iterate (stepOnce y x') initTheta
>       thetaUpdate theta = V.foldl (\theta (y, x) -> stepOnceStoch y x theta) theta $ V.zip y x'
>       finalTheta1 = V.foldl (\theta (y, x) -> stepOnceStoch y x theta) initTheta $ V.zip y x'
>       finalTheta2 = V.foldl (\theta (y, x) -> stepOnceStoch y x theta) finalTheta1 $ V.zip y x'
>       finalTheta3 = V.foldl (\theta (y, x) -> stepOnceStoch y x theta) finalTheta2 $ V.zip y x'
>       finalTheta4 = V.foldl (\theta (y, x) -> stepOnceStoch y x theta) finalTheta3 $ V.zip y x'
>
>   putStrLn $ show $ take 1 $ drop nIters hs
>   putStrLn $ show $ finalTheta1
>   putStrLn $ show $ finalTheta2
>   putStrLn $ show $ finalTheta3
>   putStrLn $ show $ finalTheta4
>   let f = foldr (.) id $ replicate 20 thetaUpdate
>   putStrLn $ show $ f initTheta




