% Logistic Regression and Automated Differentiation
% Dominic Steinitz
% 27th April 2013

Introduction
------------

Having shown how to us automated differentiation to estimate
parameters in the case of linear regression let us now turn our
attention to the problem of classification. For example, we might have
some data about people's social networking such as volume of twitter
interactions and number of twitter followers together with a label
which represents a human judgement about which one of the two
individuals is more influential. We would like to predict, for a pair
of individuals, the human judgement on who is more influential.

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

In order to maximize the cost function, we use the method of steepest
ascent: if $\boldsymbol{\theta}^i$ is a guess for the parameters of
the model then we can improve the guess by stepping a small distance
in the direction of greatest change.

We now use [gradient descent][GradientDescent] to find the maximum by
starting with a random value for the unknown parameter and then
stepping in the steepest direction.

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


  [GradientDescent]: http://en.wikipedia.org/wiki/Gradient_descent
  [StochasticGradientDescent]: http://en.wikipedia.org/wiki/Stochastic_gradient_descent

Implementation
--------------

Some pragmas to warn us about potentially dangerous situations.

> {-# OPTIONS_GHC -Wall                    #-}
> {-# OPTIONS_GHC -fno-warn-name-shadowing #-}
> {-# OPTIONS_GHC -fno-warn-type-defaults  #-}

> module Logistic (main) where

Modules from the automatic differentiation [library][Package:ad].

  [Package:ad]: http://hackage.haskell.org/package/ad-3.4

> import Numeric.AD
> import Numeric.AD.Types

> import qualified Data.Vector as V
> import Control.Monad

Some modules from a random number generator [library][Package:random-fu] as we will want to
generate some test data.

  [Package:random-fu]: http://hackage.haskell.org/package/random-fu-0.2.4.0

> import Data.Random ()
> import Data.Random.Distribution.Uniform
> import Data.Random.Distribution.Bernoulli
> import Data.RVar

Our model: the probability that $y$ has the label 1 given the observations $\boldsymbol{x}$.

> logit :: Floating a =>
>          V.Vector a ->
>          V.Vector a -> a
> logit x theta = 1 / (1 + exp (negate $ V.sum $ V.zipWith (*) theta x))

For each observation, the log likelihood:

> logLikelihood :: Floating a => V.Vector a -> a -> V.Vector a -> a
> logLikelihood theta y x = y * log (logit x theta) + (1 - y) * log (1 - logit x theta)

> totalLogLikelihood :: Floating a =>
>                       V.Vector a ->
>                       V.Vector a ->
>                       V.Vector (V.Vector a) ->
>                       a
> totalLogLikelihood theta y x = V.sum $ V.zipWith (logLikelihood theta) y x
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

Let's try it out. First we need to generate some data.

> createSample :: Double -> V.Vector Double -> IO (Double, V.Vector Double)
> createSample range theta = do
>   let l = V.length theta
>   x <- liftM (V.cons 1.0) $
>        V.sequence $
>        V.replicate (l - 1) $
>        sampleRVar $
>        uniform (negate range) range
>   putStrLn $ show x
>   putStrLn $ show $ logit x theta
>   -- isCorrectlyClassified <- sampleRVar $
>   --                          bernoulli $
>   --                          logit x theta
>   let foo = fromIntegral $ fromEnum (logit x theta > 0.5)
>   putStrLn $ show foo
>   return (foo {- isCorrectlyClassified -}, x)

We create a model with two independent variables and thus three parameters.

> actualTheta :: V.Vector Double
> actualTheta = V.fromList [0.0, 1.0]

We initialise our algorithm with arbitrary values.

> initTheta :: V.Vector Double
> initTheta = V.replicate (V.length actualTheta) 0.1
>
> nSamples :: Int
> nSamples = 10
>
> gamma :: Double
> gamma = 0.01

Now we can run our example. For the constant parameter of our model
(aka in machine learning as the bias) we ensure that the correspoding
"independent variable" is always set to $1.0$.

> main :: IO ()
> main = do
>   vals <- V.sequence $ V.replicate nSamples $ createSample 1.0 actualTheta
>   -- putStrLn $ show vals
>   let u = V.map fst vals
>       v = V.map snd vals
>       w = estimates u v initTheta
>   -- putStrLn $ show $ take 10 w
>   let hs = iterate (stepOnce gamma u v) initTheta
>   putStrLn $ show $ take 10 $ drop 100 hs
