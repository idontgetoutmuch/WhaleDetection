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

  [StochasticGradientDescent]: http://en.wikipedia.org/wiki/Stochastic_gradient_descent

We can define a cost function:

$$
\mathcal{J}(\boldsymbol{\theta}) = \frac{1}{2n}\sum_{i=1}^n (y^{(i)} - \boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)})^2
$$

Clearly minimizing this will give the same result. The constant $1/2$ is to make the manipulation of the derivative easier. In our case, this is irrelevant as we are not going to derive the derivative explicitly but use automated differentiation.

In order to mininize the cost function, we use the method of steepest ascent (or in this case descent): if $\boldsymbol{\theta}^i$ is a guess for the parameters of the model then we can improve the guess by stepping a small distance in the direction of greatest change.

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

Implementation
--------------

Some pragmas to warn us about potentially dangerous situations.

> {-# OPTIONS_GHC -Wall                    #-}
> {-# OPTIONS_GHC -fno-warn-name-shadowing #-}
> {-# OPTIONS_GHC -fno-warn-type-defaults  #-}

> module Linear where

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

Our model: the predicted value of $y$ is $\hat{y}$ given the observations $\boldsymbol{x}$.

> yhat :: Floating a =>
>         V.Vector a ->
>         V.Vector a -> a
> yhat x theta = V.sum $ V.zipWith (*) theta x

For each observation, the "cost" of the difference between the actual
value of $y$ and its predicted value.

> cost :: Floating a =>
>         V.Vector a ->
>         a ->
>         V.Vector a
>         -> a
> cost theta y x = 0.5 * (y - yhat x theta)^2

To find its gradient we merely apply the operator `grad`.

> delCost :: Floating a =>
>            a ->
>            V.Vector a ->
>            V.Vector a ->
>            V.Vector a
> delCost y x = grad $ \theta -> cost theta (auto y) (V.map auto x)

We can use the single observation cost function to define the total cost function.

> totalCost :: Floating a =>
>              V.Vector a ->
>              V.Vector a ->
>              V.Vector (V.Vector a)
>              -> a
> totalCost theta y x = (/l) $ V.sum $ V.zipWith (cost theta) y x
>   where
>     l = fromIntegral $ V.length y

Again taking the derivative is straightforward.

> delTotalCost :: Floating a =>
>                 V.Vector a ->
>                 V.Vector (V.Vector a) ->
>                 V.Vector a ->
>                 V.Vector a
> delTotalCost y x = grad f
>   where
>     f theta = totalCost theta (V.map auto y) (V.map (V.map auto) x)

Now we can implement steepest descent.

> stepOnce :: Double ->
>             V.Vector Double ->
>             V.Vector (V.Vector Double) ->
>             V.Vector Double ->
>             V.Vector Double
> stepOnce gamma y x theta =
>   V.zipWith (-) theta (V.map (* gamma) $ del theta)
>   where
>     del = delTotalCost y x

> stepOnceStoch :: Double ->
>                  Double ->
>                  V.Vector Double ->
>                  V.Vector Double ->
>                  V.Vector Double
> stepOnceStoch gamma y x theta =
>   V.zipWith (-) theta (V.map (* gamma) $ del theta)
>   where
>     del = delCost y x

Let's try it out. First we need to generate some data.

> createSample :: Double -> V.Vector Double -> IO (Double, V.Vector Double)
> createSample sigma2 theta = do
>   let l = V.length theta
>   x <- V.sequence $ V.replicate (l - 1) $ sampleRVar stdUniform
>   let mu = (theta V.! 0) + yhat x (V.drop 1 theta)
>   y <- sampleRVar $ normal mu sigma2
>   return (y, x)

We create a model with two independent variables and thus three parameters.

> actualTheta :: V.Vector Double
> actualTheta = V.fromList [0.0, 0.6, 0.7]

We initialise our algorithm with arbitrary values.

> initTheta :: V.Vector Double
> initTheta = V.replicate 3 0.1

We give our model an arbitrary variance.

> sigma2 :: Double
> sigma2 = 0.01

And set the learning rate and the number of iterations.

> nSamples, nIters:: Int
> nSamples = 100
> nIters = 2000
> gamma :: Double
> gamma = 0.1

Now we can run our example. For the constant parameter of our model
(aka in machine learning as the bias) we ensure that the correspoding
"independent variable" is always set to $1.0$.

> main :: IO ()
> main = do
>   vals <- V.sequence $ V.replicate nSamples $ createSample sigma2 actualTheta
>   let y = V.map fst vals
>       x = V.map snd vals
>       x' =  V.map (V.cons 1.0) x
>       hs = iterate (stepOnce gamma y x') initTheta
>       update theta = V.foldl (\theta (y, x) -> stepOnceStoch gamma y x theta) theta $
>                      V.zip y x'
>   putStrLn $ show $ take 1 $ drop nIters hs
>   let f = foldr (.) id $ replicate nSamples update
>   putStrLn $ show $ f initTheta

And we get quite reasonable estimates for the parameter.

    [ghci]
    main


