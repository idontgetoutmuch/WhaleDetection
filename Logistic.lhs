% Regression and Automated Differentiation
% Dominic Steinitz
% 24th April 2013

Introduction
------------

Automated differentiation was developed in the 1960's but even now
does not seem to be that widely used. Even experienced and
knowledgeable practitioners often assume it is either a finite
difference method or symbolic computation when it is neither.

  [Backpropagation]: http://en.wikipedia.org/wiki/Backpropagation
  [AutomaticDifferentiation]: http://en.wikipedia.org/wiki/Automatic_differentiation
  [Domke2009a]: http://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/

This article gives a very simple application of it in a machine
learning / statistics context.

Haskell Foreword
----------------

Some pragmas and imports required for the example code.

> {-# OPTIONS_GHC -Wall                    #-}
> {-# OPTIONS_GHC -fno-warn-name-shadowing #-}
> {-# OPTIONS_GHC -fno-warn-type-defaults  #-}

> module Main (main) where



> import Numeric.LinearAlgebra
> import Numeric.AD
> import Numeric.AD.Types
> import Numeric.GSL.Fitting.Linear

> import Data.Csv hiding (Field)
> import qualified Data.ByteString.Lazy as BS
> import System.IO
> import Data.Char
> import qualified Data.Vector as V

Multivariate Linear Logistic Regression
---------------------------------------

Let us consider linear regression first. We have

$$
\vec{y} = X\vec{w}
$$

We wish to minimise the loss function:

$$
\mathbb{J} = \vec{r}\cdot\vec{r}
$$

where

$$
\vec{r} = \vec{y} - X\vec{w}
$$

Differentiating:

$$
\frac{\partial\mathbb{L}}{\partial w_j} = 2 \sum_{i=1}^n r_i \frac{\partial r_i}{\partial w_j}
$$

We also have that:

$$
\frac{\partial r_i}{\partial w_j} = -X_{ij}
$$

Substituting:

$$
\frac{\partial\mathbb{L}}{\partial w_j} = 2 \sum_{i=1}^n (y_i - \sum_{k=1}^m X_{ik}w_k)(-X_{ij})
$$

The minimum of the loss function is reached when

$$
\frac{\partial \mathbb{L}}{\partial w_j} = 0
$$

Substituting again we have find that the values of $\vec{w} = \vec{\hat{w}}$ which
minimise the loss function satisfy

$$
2 \sum_{i=1}^n (y_i - \sum_{k=1}^m X_{ik}\hat{w}_k)(-X_{ij}) = 0
$$


  [LogisticRegression]: http://en.wikipedia.org/wiki/Logistic_regression

FIXME: We should reference the GLM book.

> myOptions :: DecodeOptions
> myOptions = defaultDecodeOptions {
>   decDelimiter = fromIntegral (ord ',')
>   }

> gamma :: Double
> gamma = 0.01

> yhat :: Floating a => V.Vector a -> V.Vector a -> a
> yhat x theta = V.sum $ V.zipWith (*) theta x
>
> cost :: Floating a => V.Vector a -> a -> V.Vector a -> a
> cost theta y x = 0.5 * (y - yhat x theta)^2
>
> totalCost :: Floating a => V.Vector a -> V.Vector a -> V.Vector (V.Vector a) -> a
> totalCost theta y x = (/l) $ V.sum $ V.zipWith (cost theta) y x
>   where
>     l = fromIntegral $ V.length y

> delCost :: forall a. Floating a => a -> V.Vector a -> V.Vector a -> V.Vector a
> delCost y x = grad $ \theta -> cost theta (auto y) (V.map auto x)
>
> delTotalCost :: forall a. Floating a => V.Vector a -> V.Vector (V.Vector a) -> V.Vector a -> V.Vector a
> delTotalCost y x = grad $ \theta -> totalCost theta (V.map auto y) (V.map (V.map auto) x)

Although we only have two independent variables, we need three
parameters for the model.

> initTheta :: V.Vector Double
> initTheta = V.replicate 3 0.1

> linReg :: IO ()
> linReg = do
>   vals <- withFile "LinRegData.csv" ReadMode
>           (\h -> do c <- BS.hGetContents h
>                     let mvv :: Either String (V.Vector (V.Vector Double))
>                         mvv = decodeWith myOptions True c
>                     case mvv of
>                       Left s -> do putStrLn s
>                                    return V.empty
>                       Right vv -> return vv
>           )
>
>   let y     = V.map (V.! 0) vals
>       x     = V.map (V.drop 1) vals
>       x'    = V.map (V.cons 1.0) x
>
>       g theta (y, x) = V.zipWith (-) theta (V.map (* gamma) $ delCost y x theta)
>       bar = V.foldl g initTheta (V.zip y x')
>       h theta = V.zipWith (-) theta (V.map (* gamma) $ del theta)
>         where
>           del = delTotalCost y x'
>       hs = iterate h initTheta
>
>   putStrLn $ show y
>   putStrLn $ show x'
>   putStrLn ""
>   putStrLn $ show $ take 10 $ drop 1000 hs
>   putStrLn ""
>   putStrLn $ show bar
>
>   let yHmat = fromList $ V.toList y
>       rowsV = V.map (V.toList) x
>       nRows = V.length rowsV
>       nCols = V.length $ V.head x
>       xHmat = (><) nRows nCols $ concat $ V.toList rowsV
>
>       (coeffs, covMat, _) = multifit xHmat yHmat
>       ests =  V.map (\x -> fst $ multifit_est (fromList x) coeffs covMat) rowsV
>       diffs = zipWith (-) (toList yHmat) (V.toList ests)
>   putStrLn $ show $ (sum (map (^2) diffs) / (fromIntegral $ length diffs))
>   putStrLn $ show coeffs
>   putStrLn ""
>   putStrLn "END LINEAR"
>   putStrLn ""

FIXME: Reference for neural net, multi-layer perceptron and logistic
regression.

We can view neural nets or at least a multi layer perceptron as a
generalisation of (multivariate) linear logistic regression. It is
instructive to apply both backpropagation and automated
differentiation to this simpler problem.

Following [Ng][Ng:cs229], we define:

  [Ng:cs229]: http://cs229.stanford.edu

$$
h_{\theta}(\vec{x}) = g(\theta^T\vec{x})
$$

where $\theta = (\theta_1, \ldots, \theta_m)$ and $g$ is a function
such as the logistic function $g(x) = 1 / (1 + e^{-\theta^T\vec{x}})$
or $\tanh$.

$$
y = tanh (\sum_{i=1}^{22} w_i * x_i + c)
$$

Next we define the probability of getting a particular value of the binary lable:

$$
\begin{align*}
{\mathbb P}(y = 1 \mid \vec{x}; \theta) &= h_{\theta}(\vec{x}) \\
{\mathbb P}(y = 0 \mid \vec{x}; \theta) &= 1 - h_{\theta}(\vec{x})
\end{align*}
$$

which we can re-write as:

$$
p(y \mid \vec{x} ; \theta) = (h_{\theta}(\vec{x}))^y(1 - h_{\theta}(\vec{x}))^{1 - y}
$$

We wish to find the value of $\theta$ that gives the maximum
probability to the observations. We do this by maximising the
likelihood. Assuming we have $n$ observations the likelihood is:

$$
\begin{align*}
L(\theta) &= \prod_{i=1}^n p(y^{(i)} \mid {\vec{x}}^{(i)} ; \theta) \\
          &= \prod_{i=1}^n (h_{\theta}(\vec{x}^{(i)}))^{y^{(i)}} (1 - h_{\theta}(\vec{x}^{(i)}))^{1 - y^{(i)}}
\end{align*}
$$

It is standard practice to maximise the log likelihood which will give the same maximum as log is monotonic.

$$
\begin{align*}
l(\theta) &= \log L(\theta) \\
          &= \sum_{i=1}^n {y^{(i)}}\log h_{\theta}(\vec{x}^{(i)}) + (1 - y^{(i)})\log (1 - h_{\theta}(\vec{x}^{(i)}))
\end{align*}
$$

We now use [gradient descent][GradientDescent] to find the maximum by
starting with a random value for the unknown parameter and then
stepping in the steepest direction.

  [GradientDescent]: http://en.wikipedia.org/wiki/Gradient_descent

$$
\theta' = \theta + \gamma\nabla_{\theta}l(\theta)
$$

Differentiating the log likelihood, we have:

$$
\begin{align*}
\frac{\partial}{\partial \theta_i}l(\theta) &= \big(y\frac{1}{g(\theta^T\vec{x})} - (1 - y)\frac{1}{1 - g(\theta^T\vec{x})}\big)\frac{\partial}{\partial \theta_i}g(\theta^T\vec{x})
\end{align*}
$$

> logit :: Floating a => V.Vector a -> V.Vector a -> a
> logit x theta = 1 / (1 + exp (V.sum $ V.zipWith (*) theta x))
>
> logLikelihood :: Floating a => V.Vector a -> a -> V.Vector a -> a
> logLikelihood theta l x = l * log (logit x theta) + (1 - l) * log (1 - logit x theta)
>
> initWs :: V.Vector Double
> initWs = V.replicate 11 0.1
>
> logReg :: IO ()
> logReg = do
>   vals <- withFile "/Users/dom/Downloadable/DataScienceLondon/Train.csv" ReadMode
>           (\h -> do c <- BS.hGetContents h
>                     let mvv :: Either String (V.Vector (V.Vector Double))
>                         mvv = decodeWith myOptions True c
>                     case mvv of
>                       Left s -> do putStrLn s
>                                    return V.empty
>                       Right vv -> return vv
>           )
>   let labels = V.map (V.! 0) vals
>       inds  = V.map (V.drop 1) vals
>       featuress = V.map (V.splitAt 11) inds
>       tF x = log $ x + 1
>       xTrain = V.map (\fs -> V.zipWith (-) (V.map tF $ fst fs) (V.map tF $ snd fs))
>                      featuress
>       delLogLikelihood l x = grad $
>                              \theta -> logLikelihood theta (auto l) (V.map auto x)
>       g theta (l, x) = V.zipWith (+) theta
>                        (V.map (* gamma) $ delLogLikelihood l x theta)
>       bar = V.foldl g initWs (V.zip (V.take 2000 labels) (V.take 2000 xTrain))
>       baz = V.foldl g initWs (V.zip (V.take 2010 labels) (V.take 2010 xTrain))
>   putStrLn $ show $ initWs
>   putStrLn $ show $ bar
>   putStrLn $ show $ V.zipWith (-) bar baz
>   putStrLn $ show $ V.maximum $ V.zipWith (-) bar baz



```{.dia width='400'}
import NnClassifierDia
dia = nn
```

> main :: IO ()
> main = do
>   linReg
>   logReg


