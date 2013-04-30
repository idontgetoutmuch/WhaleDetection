% Logistic Regression and Automated Differentiation
% Dominic Steinitz
% 27th April 2013

Introduction
------------

Having shown how to use automated differentiation to estimate
parameters in the case of [linear regression][blog:linearRegression]
let us now turn our attention to the problem of classification. For
example, we might have some data about people's social networking such
as volume of twitter interactions and number of twitter followers
together with a label which represents a human judgement about which
one of the two individuals is more influential. We would like to
predict, for a pair of individuals, the human judgement on who is more
influential.

  [blog:linearRegression]: http://idontgetoutmuch.wordpress.com/2013/04/26/regression-and-automated-differentiation-4

Logistic Regression
------------------------------

We define the probability of getting a particular value of the binary label:

$$
\begin{aligned}
{\mathbb P}(y = 1 \mid \boldsymbol{x}; \boldsymbol{\theta}) &= h_{\boldsymbol{\theta}}(\boldsymbol{x}) \\
{\mathbb P}(y = 0 \mid \boldsymbol{x}; \boldsymbol{\theta}) &= 1 - h_{\boldsymbol{\theta}}(\boldsymbol{x})
\end{aligned}
$$

where $\boldsymbol{x^{(i)}}$ and $\boldsymbol{\theta}$ are column vectors of size $m$

$$
h_{\boldsymbol{\theta}}(\boldsymbol{x}) = g(\boldsymbol{\theta}^T\boldsymbol{x})
$$

and $g$ is a function
such as the logistic function $g(x) = 1 / (1 + e^{-x})$
or $\tanh$.

We can re-write this as:

$$
p(y \mid \boldsymbol{x} ; \boldsymbol{\theta}) = (h_{\boldsymbol{\theta}}(\boldsymbol{x}))^y(1 - h_{\boldsymbol{\theta}}(\boldsymbol{x}))^{1 - y}
$$

We wish to find the value of $\boldsymbol{\theta}$ that gives the maximum
probability to the observations. We do this by maximising the
likelihood. Assuming we have $n$ observations the likelihood is:

$$
\begin{align*}
\mathcal{L}(\boldsymbol{\theta}) &= \prod_{i=1}^n p(y^{(i)} \mid {\boldsymbol{x}}^{(i)} ; \boldsymbol{\theta}) \\
          &= \prod_{i=1}^n (h_{\boldsymbol{\theta}}(\boldsymbol{x}^{(i)}))^{y^{(i)}} (1 - h_{\boldsymbol{\theta}}(\boldsymbol{x}^{(i)}))^{1 - y^{(i)}}
\end{align*}
$$

It is standard practice to maximise the log likelihood which will give the same maximum as log is monotonic.

$$
\begin{align*}
\lambda(\boldsymbol{\theta}) &= \log \mathcal{L}(\boldsymbol{\theta}) \\
          &= \sum_{i=1}^n {y^{(i)}}\log h_{\boldsymbol{\theta}}(\boldsymbol{x}^{(i)}) + (1 - y^{(i)})\log (1 - h_{\boldsymbol{\theta}}(\boldsymbol{x}^{(i)}))
\end{align*}
$$

In order to maximize the cost function, we again use the method of [steepest
ascent][GradientDescent]: if $\boldsymbol{\theta}^i$ is a guess for the parameters of
the model then we can improve the guess by stepping a small distance
in the direction of greatest change.

$$
\boldsymbol{\theta}^{i+1} = \boldsymbol{\theta}^{i} - \gamma \nabla\mathcal{J}(\boldsymbol{\theta})
$$

$\gamma$ is some constant known in machine learning as the learning
rate. It must be chosen to be large enough to ensure convergence
within a reasonable number of steps but not so large that the
algorithm fails to converge.

When the number of observations is high then the cost of evaluating
the cost function can be high; as a cheaper alternative we can use
[stochastic gradient descent][StochasticGradientDescent]. Instead of
taking the gradient with respect to all the observations, we take the
gradient with respect to each observation in our data set. Of course
if our data set is small we may have to use the data set several times
to achieve convergence.

When the observations / training data are linearly separable then the
magnitude of the parameters can grow without bound as the
(parametized) logistic function then tends to the Heaviside / step
function. Moreover, it is obvious that there can be more than one
separaing hyperplane in this circumstance. To circumvent these
infelicities, we instead maximize a penalized log likelihood
function:

$$
\sum_{i=1}^n {y^{(i)}}\log h_{\boldsymbol{\theta}}(\boldsymbol{x}^{(i)}) + (1 - y^{(i)})\log (1 - h_{\boldsymbol{\theta}}(\boldsymbol{x}^{(i)})) - \frac{\delta}{2}\|\boldsymbol{\theta}\|^2
$$

See [Bishop][bishop:ml] and [Mitchell][mitchell:ml] for further details.

  [GradientDescent]: http://en.wikipedia.org/wiki/Gradient_descent
  [StochasticGradientDescent]: http://en.wikipedia.org/wiki/Stochastic_gradient_descent
  [bishop:ml]: http://research.microsoft.com/en-us/um/people/cmbishop/prml/
  [mitchell:ml]: http://www.cs.cmu.edu/%7Etom/mlbook.html

Implementation
--------------

Some pragmas to warn us about potentially dangerous situations.

FIXME: Replace the pragmas!!!

> {-# LANGUAGE TupleSections #-}
>
> module Logistic ( betas
>                 , main
>                 , a
>                 , b
>                 , nSamples
>                 ) where

Modules from the automatic differentiation [library][Package:ad].

  [Package:ad]: http://hackage.haskell.org/package/ad-3.4

> import Numeric.AD
> import Numeric.AD.Types

> import qualified Data.Vector as V
> import Control.Monad
> import Control.Monad.State
> import Data.List
> import Text.Printf

Some modules from a random number generator [library][Package:random-fu] as we will want to
generate some test data.

  [Package:random-fu]: http://hackage.haskell.org/package/random-fu-0.2.4.0

> import System.Random
> import Data.Random ()
> import Data.Random.Distribution.Uniform
> import Data.Random.Distribution.Beta
> import Data.RVar

Our model: the probability that $y$ has the label 1 given the observations $\boldsymbol{x}$.

> logit :: Floating a =>
>          a -> a
> logit x = 1 / (1 + exp (negate x))

For each observation, the log likelihood:

> logLikelihood :: Floating a => V.Vector a -> a -> V.Vector a -> a
> logLikelihood theta y x = y * log (logit z) + (1 - y) * log (1 - logit z)
>   where
>     z = V.sum $ V.zipWith (*) theta x

> totalLogLikelihood :: Floating a =>
>                       V.Vector a ->
>                       V.Vector a ->
>                       V.Vector (V.Vector a) ->
>                       a
> totalLogLikelihood theta y x = a - delta * b
>   where
>     l = fromIntegral $ V.length y
>     a = V.sum $ V.zipWith (logLikelihood theta) y x
>     b = (/2) $ sqrt $ V.sum $ V.map (^2) theta
>
> estimates :: (Floating a, Ord a) =>
>              V.Vector a ->
>              V.Vector (V.Vector a) ->
>              V.Vector a ->
>              [V.Vector a]
> estimates y x = gradientAscent $
>                 \theta -> totalLogLikelihood theta (V.map auto y) (V.map (V.map auto) x)
> delTotalLogLikelihood :: Floating a =>
>                 V.Vector a ->
>                 V.Vector (V.Vector a) ->
>                 V.Vector a ->
>                 V.Vector a
> delTotalLogLikelihood y x = grad f
>   where
>     f theta = totalLogLikelihood theta (V.map auto y) (V.map (V.map auto) x)
>
> stepOnce :: Double ->
>             V.Vector Double ->
>             V.Vector (V.Vector Double) ->
>             V.Vector Double ->
>             V.Vector Double
> stepOnce gamma y x theta = V.zipWith (+) theta (V.map (* gamma) $ del theta)
>   where
>     del = delTotalLogLikelihood y x

To find its gradient we merely apply the operator `grad`.

Now we can implement steepest descent.

Let's try it out. First we need to generate some data.  Rather
arbitrarily let us create some populations from the `beta`
distribution.

> betas :: Int -> Double -> Double -> [Double]
> betas n a b =
>   fst $ runState (replicateM n (sampleRVar (beta a b))) (mkStdGen seed)
>     where
>       seed = 0

We can plot the populations we wish to distinguish by sampling.

> a          = 15
> b          = 6
> nSamples   = 100000

> sample0 = betas nSamples a b
> sample1 = betas nSamples b a

Note that in this case we could come up with a classification rule by
inspecting the histograms. Furthermore, the populations overlap which
means we will inevitably mis-classify some observations.

```{.dia width='800'}
{-# LANGUAGE TupleSections #-}

import Diagrams.Prelude
import Data.Colour (withOpacity)
import Logistic

import Data.Random ()
import Data.Random.Distribution.Beta
import Data.RVar

import System.Random

import Data.List
import qualified Data.IntMap as IntMap

import Control.Monad.State

import Text.Printf

tickSize   = 0.1
nCells     = 100
cellColour0 = red  `withOpacity` 0.5
cellColour1 = blue `withOpacity` 0.5

background = rect 1.1 1.1 # translate (r2 (0.5, 0.5))

test tickSize nCells a0 b0 a1 b1 nSamples =
  ticks [0.0, tickSize..1.0] <>
  hist cellColour0 xs <>
  hist cellColour1 ys <>
  background
    where
      xs = IntMap.elems $
      	   IntMap.map fromIntegral $
           histogram nCells $
           betas nSamples a0 b0
      ys = IntMap.elems $
      	   IntMap.map fromIntegral $
           histogram nCells $
           betas nSamples a1 b1

hist cellColour xs = scaleX sX . scaleY sY . position $ hist' where
    ysmax = fromInteger . ceiling $ maximum xs
    ysmin = fromInteger . floor $ minimum xs
    xsmax = fromIntegral $ length xs
    xsmin = 0.0
    sX = 1 / (xsmax - xsmin)
    sY = 1 / (ysmax - ysmin)
    hist' = zip (map p2 $ map (,0) $
            map fromInteger [0..]) (map (cell 1) xs)
    cell w h = alignB $ rect w h
                      # fcA cellColour
                      # lc white
                      # lw 0.001

ticks xs = (mconcat $ map tick xs)  <> line
  where
    maxX   = maximum xs
    line   = fromOffsets [r2 (maxX, 0)]
    tSize  = maxX / 100
    tick x = endpt # translate tickShift
      where
        tickShift = r2 (x, 0)
        endpt     = topLeftText (printf "%.2f" x) # fontSize (tSize * 2) <>
                    circle tSize # fc red  # lw 0

histogram :: Int -> [Double] -> IntMap.IntMap Int
histogram nCells xs =
  foldl' g emptyHistogram xs
    where
      g h x          = IntMap.insertWith (+) (makeCell nCells x) 1 h
      emptyHistogram = IntMap.fromList $ zip [0 .. nCells - 1] (repeat 0)
      makeCell m     = floor . (* (fromIntegral m))

dia = test tickSize nCells a b b a nSamples
```

> mixSamples :: [Double] -> [Double] -> [(Double, Double)]
> mixSamples xs ys = unfoldr g ((map (0,) xs), (map (1,) ys))
>   where
>     g ([], [])   = Nothing
>     g ((x:xs), ys) = Just $ (x, (ys, xs))

> createSample :: V.Vector (Double, Double)
> createSample = V.fromList $ take 100 $ mixSamples sample1 sample0

We create a model with one independent variables and thus two parameters.

> actualTheta :: V.Vector Double
> actualTheta = V.fromList [0.0, 1.0]

We initialise our algorithm with arbitrary values.

> initTheta :: V.Vector Double
> initTheta = V.replicate (V.length actualTheta) 0.1

Set the learning rate, the strength of the penalty term and the number
of iterations.

> gamma :: Double
> gamma = 0.04
>
> delta :: Floating a => a
> delta = 1.0
>
> nIters :: Int
> nIters = 4000

Now we can run our example. For the constant parameter of our model
(aka in machine learning as the bias) we ensure that the correspoding
"independent variable" is always set to $1.0$.

> vals :: V.Vector (Double, V.Vector Double)
> vals = V.map (\(y, x) -> (y, V.fromList [1.0, x])) $ createSample

> main :: IO ()
> main = do
>   let u = V.map fst vals
>       v = V.map snd vals
>       hs = iterate (stepOnce gamma u v) initTheta
>       xs = V.map snd vals
>       theta = head $ drop nIters hs
>   printf "theta_0 = %5.2f, theta_1 = %5.2f\n" (theta V.! 0) (theta V.! 1)
>   let predProbs  = V.map (\x -> logit $ V.sum $ V.zipWith (*) theta x) xs
>       mismatches = V.filter (> 0.5) $
>                    V.map abs $
>                    V.zipWith (-) actuals preds
>         where
>           actuals = V.map fst vals
>           preds   = V.map (\x -> fromIntegral $ fromEnum (x > 0.5)) predProbs
>   let lActuals, lMisMatches :: Double
>       lActuals    = fromIntegral $ V.length vals
>       lMisMatches = fromIntegral $ V.length mismatches
>   printf "%5.2f%% correct\n" $ 100.0 *  (lActuals - lMisMatches) / lActuals

And we get quite reasonable estimates:

    [ghci]
    main
