% Backpropogation is Just Steepest Descent with Automatic Differentiation
% Dominic Steinitz
% 20th September 2013

Preface
=======

The intended audience of this article is someone who knows something
about Machine Learning and Artifical Neural Networks (ANNs) in
particular and who recalls that fitting an ANN required a technique
called backpropagation. The goal of this post is to refresh the
reader's knowledge of ANNs and backpropagation and to show that the
latter is merely a specialised version of *automatic
differentiation*, a tool that all Machine Learning practitioners
should know about and have in their toolkit.

Introduction
============

The problem is simple to state: we have a (highly) non-linear
function, the cost function of an Artificial Neural Network (ANN), and
we wish to minimize this so as to estimate the parameters / weights of
the function.

In order to minimise the function, one obvious approach is to use
steepest descent: start with random values for the parameters to be
estimated, find the direction in which the the function decreases most
quickly, step a small amount in that direction and repeat until close
enough.

But we have two problems:

* We have an algorithm or a computer program that calculates the
non-linear function rather than the function itself.

* The function has a very large number of parameters, hundreds if not
thousands.

One thing we could try is bumping each parameter by a small amount to
get partial derivatives numerically

$$
\frac{\partial E(\ldots, w, \ldots)}{\partial w} \approx \frac{E(\ldots, w + \epsilon, \ldots) - E(\ldots, w, \ldots)}{\epsilon}
$$

But this would mean evaluating our function many times and moreover we
could easily get numerical errors as a result of the vagaries of
floating point arithmetic.

As an alternative we could turn our algorithm or computer program into
a function more recognisable as a mathematical function and then
compute the differential itself as a function either by hand or by
using a symbolic differentiation package. For the complicated
expression which is our mathematical function, the former would be
error prone and the latter could easily generate something which would
be even more complex and costly to evaluate than the original
expression.

The standard approach is to use a technique called
[backpropagation](http://en.wikipedia.org/wiki/Backpropagation) and
the understanding and application of this technique forms a large part
of many machine learning lecture courses.

Since at least the 1960s techniques for automatically differentiating
computer programs have been discovered and re-discovered. Anyone who
knows about these techniques and reads about backpropagation quickly
realises that backpropagation is just automatic differentiation and
steepest descent.

This article is divided into

 * Refresher on neural networks and backpropagation;

 * Methods for differentiation;

 * Backward and forward automatic differentiation and

 * Concluding thoughts.

The only thing important to remember throughout is the chain rule

$$
(g \circ f)'(a) = g'(f(a))\cdot f'(a)
$$

in alternative notation

$$
\frac{\mathrm{d} (g \circ f)}{\mathrm{d} x}(a) =
\frac{\mathrm{d} g}{\mathrm{d} y}(f(a)) \frac{\mathrm{d} f}{\mathrm{d} x}(a)
$$

where $y = f(x)$. More suggestively we can write

$$
\frac{\mathrm{d} g}{\mathrm{d} x} =
\frac{\mathrm{d} g}{\mathrm{d} y} \frac{\mathrm{d} y}{\mathrm{d} x}
$$

where it is understood that $\mathrm{d} g / \mathrm{d} x$ and
$\mathrm{d} y / \mathrm{d} x$ are evaluated at $a$ and $\mathrm{d} g /
\mathrm{d} y$ is evaluated at $f(a)$.

For example,

$$
\frac{\mathrm{d}}{\mathrm{d} x} \sqrt{3 \sin(x)} =
\frac{\mathrm{d}}{\mathrm{d} x} (3 \sin(x)) \cdot \frac{\mathrm{d}}{\mathrm{d} y} \sqrt{y} =
3 \cos(x) \cdot \frac{1}{2\sqrt{y}} =
\frac{3\cos(x)}{2\sqrt{3\sin(x)}}
$$

Acknowledgements
================

Sadly I cannot recall all the sources I looked at in order to produce
this article but I have made heavy use of the following.

* Justin Domke's [blog](http://justindomke.wordpress.com)

* Conal Elliot's [website](http://conal.net/papers/beautiful-differentiation)

* Jerzy Karczmarczuk's [papers](https://karczmarczuk.users.greyc.fr/arpap)

* The [Community Portal for Automatic Differentiation](http://www.autodiff.org)

Neural Network Refresher
========================

Here is our model, with $\boldsymbol{x}$ the input,
$\hat{\boldsymbol{y}}$ the predicted output and $\boldsymbol{y}$ the
actual output and $w^{(k)}$ the weights in the $k$-th layer. We have
concretised the transfer function as $\tanh$ but it is quite popular
to use the $\text{logit}$ function.

$$
\begin{aligned}
a_i^{(1)} &= \sum_{j=0}^{N^{(1)}} w_{ij}^{(1)} x_j \\
z_i^{(1)} &= \tanh(a_i^{(1)}) \\
a_i^{(2)} &= \sum_{j=0}^{N^{(2)}} w_{ij}^{(2)} z_j^{(1)} \\
\dots     &= \ldots \\
a_i^{(L-1)} &= \sum_{j=0}^{N^{(L-1)}} w_{ij}^{(L-1)} z_j^{(L-2)} \\
z_j^{(L-1)} &= \tanh(a_j^{(L-1)}) \\
\hat{y}_i &= \sum_{j=0}^{N^{(L)}} w_{ij}^{(L)} z_j^{(L-1)} \\
\end{aligned}
$$

with the loss or cost function

$$
E(\boldsymbol{w}; \boldsymbol{x}, \boldsymbol{y}) = \frac{1}{2}\|(\hat{\boldsymbol{y}} - \boldsymbol{y})\|^2
$$

The diagram below depicts a neural network with a single hidden layer.

```{.dia width='500'}
import MLTalkDiagrams
dia = example1
```
In order to apply the steepest descent algorithm we need to calculate the differentials of this latter function with respect to the weights, that is, we need to calculate

$$
\Delta w_{ij} = \frac{\partial E}{\partial w_{ij}}
$$

Applying the chain rule

$$
\Delta w_{ij} =
\frac{\partial E}{\partial w_{ij}} =
\frac{\partial E}{\partial a_i}\frac{\partial a_i}{\partial w_{ij}}
$$

Since

$$
a_j^{(l)} = \sum_{i=0}^N w_{ij}^{(l)}z_i^{(l-1)}
$$

we have

$$
\frac{\partial a_i^{(l)}}{\partial w_{ij}^{(l)}} =
\frac{\sum_{k=0}^M w_{kj}^{(l)}z_k^{(l-1)}}{\partial w_{ij}^{(l)}} =
z_i^{(l-1)}
$$

Defining

$$
\delta_j^{(l)} \equiv
\frac{\partial E}{\partial a_j^{(l)}}
$$

we obtain

$$
\Delta w_{ij}^{(l)} =
\frac{\partial E}{\partial w_{ij}^{(l)}} =
\delta_j^{(l)} z_i^{(l-1)}
$$

Finding the $z_i$ for each layer is straightforward: we start with the
inputs and propagate forward. In order to find the $\delta_j$ we need
to start with the outputs a propagate backwards:

For the output layer we have (since $\hat{y}_j = a_j$)

$$
\delta_j = \frac{\partial E}{\partial a_j} = \frac{\partial E}{\partial y_j} = \frac{\partial}{\partial y_j}\bigg(\frac{1}{2}\sum_{i=0}^M (\hat{y}_i - y_i)^2\bigg) = \hat{y}_j - y_j
$$

For a hidden layer using the chain rule

$$
\delta_j^{(l-1)} = \frac{\partial E}{\partial a_j^{(l-1)}} =
\sum_k \frac{\partial E}{\partial a_k^{(l)}}\frac{\partial a_k^{(l)}}{\partial a_j^{(l-1)}}
$$

Now

$$
a_k^{(l)} = \sum_i w_{ki}^{(l)}z_i^{(l-1)} = \sum_i w_{ki}^{(l)} f(a_i^{(l-1)})
$$

so that

$$
\frac{\partial a_k^{(l)}}{\partial a_j^{(l-1)}} =
\frac{\sum_i w_{ki}^{(l)} f(a_i^{(l-1)})}{\partial a_j^{(l-1)}} =
w_{kj}^{(l)}\,f'(a_j^{(l-1)})
$$

and thus

$$
\delta_j^{(l-1)} =
\sum_k \frac{\partial E}{\partial a_k^{(l)}}\frac{\partial a_k^{(l)}}{\partial a_j^{(l-1)}} =
\sum_k \delta_k^{(l)} w_{kj}^{(l)}\, f'(a_j^{(l-1)}) =
f'(a_j^{(l-1)}) \sum_k \delta_k^{(l)} w_{kj}^{(l)}
$$

Summarising

1. We calculate all $a_j$ and $z_j$ for each layer starting with the
input layer and propagating forward.

2. We evaluate $\delta_j^{(L)}$ in the output layer using $\delta_j = \hat{y}_j - y_j$.

3. We evaluate $\delta_j$ in each layer using $\delta_j^{(l-1)} =
f'(a_j^{(l-1)})\sum_k \delta_k^{(l)} w_{kj}^{(l)}$ starting with the output
layer and propagating backwards.

4. Use $\partial E / \partial w_{ij}^{(l)} = \delta_j^{(l)} z_i^{(l-1)}$ to obtain the
required derivatives in each layer.

For the particular activation function $\tanh$ we have $f'(a) = \tanh'
(a) = 1 - \tanh^2(a)$. And finally we can use the partial derivatives
to step in the right direction using steepest descent

$$
w' = w - \gamma\nabla E(w)
$$

where $\gamma$ is the step size aka the learning rate.

Differentiation
===============

So now we have an efficient algorithm for differentiating the cost
function for an ANN and thus estimating its parameters but it seems
quite complex. In the introduction we alluded to other methods of
differentiation. Let us examine those in a bit more detail before
moving on to a general technique for differentiating programs of which
backpropagation turns out to be a specialisation.

Numerical Differentiation
-------------------------

Consider the function $f(x) = e^x$ then its differential $f'(x) = e^x$
and we can easily compare a numerical approximation of this with the exact
result. The numeric approximation is given by

$$
f'(x) \approx \frac{f(x + \epsilon) - f(x)}{\epsilon}
$$

In theory we should get a closer and closer approximation as epsilon
decreases but as the chart below shows at some point (with $\epsilon
\approx 2^{-26}$) the approximation worsens as a result of the fact
that we are using floating point arithmetic. For a complex function
such as one which calculates the cost function of an ANN, there is a
risk that we may end up getting a poor approximation for the
derivative and thus a poor estimate for the parameters of the model.

```{.dia width='500'}
import MLTalkDiagrams
dia = errDiag
```

Symbolic Differentiation
------------------------

Suppose we have the following program (written in Python)

~~~~ { .python }
import numpy as np

def many_sines(x):
    y = x
    for i in range(1,7):
        y = np.sin(x+y)
    return y
~~~~

When we unroll the loop we are actually evaluating

$$
f(x) = \sin(x + \sin(x + \sin(x + \sin(x + \sin(x + \sin(x + x))))))
$$

Now suppose we want to get the differential of this
function. Symbolically this would be

$$
\begin{aligned}
f'(x) &=           (((((2\cdot \cos(2x)+1)\cdot \\
      &\phantom{=} \cos(\sin(2x)+x)+1)\cdot \\
      &\phantom{=} \cos(\sin(\sin(2x)+x)+x)+1)\cdot \\
      &\phantom{=} \cos(\sin(\sin(\sin(2x)+x)+x)+x)+1)\cdot \\
      &\phantom{=} \cos(\sin(\sin(\sin(\sin(2x)+x)+x)+x)+x)+1)\cdot \\
      &\phantom{=} \cos(\sin(\sin(\sin(\sin(\sin(2x)+x)+x)+x)+x)+x)
\end{aligned}
$$

Typically the non-linear function that an ANN gives is much more
complex than the simple function given above. Thus its derivative will
correspondingly more complex and therefore expensive to
compute. Moreover calculating this derivative by hand could easily
introduce errors. And in order to have a computer perform the symbolic
calculation we would have to encode our cost function somehow so that
it is amenable to this form of manipulation.

Automatic Differentiation
=========================

Reverse Mode
------------

Traditionally, forward mode is introduced first as this is considered
easier to understand. We introduce reverse mode first as it can be
seen to be a generalization of backpropagation.

Consider the function

$$
f(x) = \exp(\exp(x) + (\exp(x))^2) + \sin(\exp(x) + (\exp(x))^2)
$$

Let us write this a data flow graph.

```{.dia width='500'}
import MLTalkDiagrams
dia = example
```

We can thus re-write our function as a sequence of simpler functions
in which each function only depends on variables earlier in the
sequence.

$$
\begin{aligned}
u_7    &= f_7(u_6, u_5, u_4, u_3, u_2, u_1) \\
u_6    &= f_6(u_5, u_4, u_3, u_2, u_1) \\
\ldots &= \ldots \\
u_1    &= f_1(u_1)
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{d}u_7    &= \frac{\partial f_7}{\partial u_6} \mathrm{d} u_6 +
                    \frac{\partial f_7}{\partial u_5} \mathrm{d} u_5 +
                    \frac{\partial f_7}{\partial u_4} \mathrm{d} u_4 +
                    \frac{\partial f_7}{\partial u_3} \mathrm{d} u_3 +
                    \frac{\partial f_7}{\partial u_2} \mathrm{d} u_2 + 
                    \frac{\partial f_7}{\partial u_1} \mathrm{d} u_1 \\
\mathrm{d}u_6    &= \frac{\partial f_6}{\partial u_5} \mathrm{d} u_5 + 
                    \frac{\partial f_6}{\partial u_4} \mathrm{d} u_4 +
                    \frac{\partial f_6}{\partial u_3} \mathrm{d} u_3 +
                    \frac{\partial f_6}{\partial u_2} \mathrm{d} u_2 +
                    \frac{\partial f_6}{\partial u_1} \mathrm{d} u_1 \\
\ldots           &= \ldots \\
\mathrm{d}u_1    &= \frac{\partial f_1}{\partial u_1} \mathrm{d} u_1
\end{aligned}
$$

In our particular example, since $u_1, \dots, u_5$ do not depend on $u_6$

$$
\begin{aligned}
\frac{\mathrm{d}u_7}{\mathrm{d}u_6} &= 1
\end{aligned}
$$

Further $u_6$ does not depend on $u_5$ so we also have

$$
\begin{aligned}
\frac{\mathrm{d}u_7}{\mathrm{d}u_5} &= 1 \\
\end{aligned}
$$

Now things become more interesting as $u_6$ and $u_5$ both depend on
$u_4$ and so the chain rule makes an explicit appearance

$$
\begin{aligned}
\frac{\mathrm{d}u_7}{\mathrm{d}u_4} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_6}\frac{\mathrm{d}u_6}{\mathrm{d}u_4} +
 \frac{\mathrm{d}u_7}{\mathrm{d}u_5}\frac{\mathrm{d}u_5}{\mathrm{d}u_4} \\
&= \frac{\mathrm{d}u_7}{\mathrm{d}u_6}\exp{u_4} +
   \frac{\mathrm{d}u_7}{\mathrm{d}u_5}\cos{u_5}
\end{aligned}
$$

Carrying on

$$
\begin{aligned}
\frac{\mathrm{d}u_7}{\mathrm{d}u_3} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_4}\frac{\mathrm{d}u_4}{\mathrm{d}u_3} \\
&= \frac{\mathrm{d}u_7}{\mathrm{d}u_4} \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_2} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_4}\frac{\mathrm{d}u_4}{\mathrm{d}u_2} +
 \frac{\mathrm{d}u_7}{\mathrm{d}u_3}\frac{\mathrm{d}u_3}{\mathrm{d}u_2} \\
&= \frac{\mathrm{d}u_7}{\mathrm{d}u_4} + 2u_2\frac{\mathrm{d}u_7}{\mathrm{d}u_4} \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_1} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_2}\frac{\mathrm{d}u_2}{\mathrm{d}u_1} \\
&=\frac{\mathrm{d}u_7}{\mathrm{d}u_2}\exp{u_2}
\end{aligned}
$$

Note that having worked from top to bottom (the forward sweep) in the
graph to calculate the function itself, we have to work backwards from
bottom to top (the backward sweep) to calculate the derivative.

So provided we can translate our program into a call graph, we can
apply this procedure to calculate the differential with the same
complexity as the original program.

The pictorial representation of an ANN is effectively the data flow
graph of the cost function (without the final cost calculation itself)
and its differential can be calculated as just being identical to
backpropagation.

Forward Mode
------------

An alternative method for automatic differentiation is called forward
mode and has a simple implementation. Let us illustrate this using
[Haskell 98].

  [Haskell 98]: http://www.haskell.org/onlinereport "haskell.org online report"

First some boilerplate declarations that need not concern us further.

> {-# LANGUAGE NoMonomorphismRestriction #-}
> 
> module AD (
>     Dual(..)
>   , f
>   , idD
>   ) where
> 
> default ()

 Let us define dual numbers

> data Dual = Dual Double Double
>   deriving (Eq, Show)

We can think of these pairs as first order polynomials in $\epsilon$,
$x + \epsilon x'$ such that $\epsilon^2 = 0$

Thus, for example, we have

$$
\begin{align}
(x + \epsilon x') + (y + \epsilon y') &= ((x + y) + \epsilon (x' + y')) \\
(x + \epsilon x')(y + \epsilon y') &= xy + \epsilon (xy' + x'y) \\
\log (x + \epsilon x') &=
\log x (1 + \epsilon \frac {x'}{x}) =
\log x + \epsilon\frac{x'}{x} \\
\sqrt{(x + \epsilon x')} &=
\sqrt{x(1 + \epsilon\frac{x'}{x})} =
\sqrt{x}(1 + \epsilon\frac{1}{2}\frac{x'}{x}) =
\sqrt{x} + \epsilon\frac{1}{2}\frac{x'}{\sqrt{x}} \\
\ldots &= \ldots
\end{align}
$$

Notice that these equations implicitly encode the chain rule. For
example, we know, using the chain rule, that

$$
\frac{\mathrm{d}}{\mathrm{d} x}\log(\sqrt x) =
\frac{1}{\sqrt x}\frac{1}{2}x^{-1/2} =
\frac{1}{2x}
$$

And using the example equations above we have

$$
\begin{align}
\log(\sqrt (x + \epsilon x')) &= \log (\sqrt{x} + \epsilon\frac{1}{2}\frac{x'}{\sqrt{x}}) \\
                              &= \log (\sqrt{x}) + \epsilon\frac{\frac{1}{2}\frac{x'}{\sqrt{x}}}{\sqrt{x}} \\
                              &= log (\sqrt{x}) + \epsilon x'\frac{1}{2x}
\end{align}
$$

Notice that dual numbers carry around the calculation and the
derivative of the calculation. To actually evaluate $\log(\sqrt{x})$
at a particular value, say 2, we plug in 2 for $x$ and 1 for $x'$

$$
\log (\sqrt(2 + \epsilon 1) = \log(\sqrt{2}) + \epsilon\frac{1}{4}
$$

Thus the derivative of $\log(\sqrt{x})$ at 2 is $1/4$.

With a couple of helper functions we can implement this rule
($\epsilon^2 = 0$) by making *Dual* an instance of *Num*, *Fractional*
and *Floating*.

> constD :: Double -> Dual
> constD x = Dual x 0
> 
> idD :: Double -> Dual
> idD x = Dual x 1.0
> 
> instance Num Dual where
>   fromInteger n             = constD $ fromInteger n
>   (Dual x x') + (Dual y y') = Dual (x + y) (x' + y')
>   (Dual x x') * (Dual y y') = Dual (x * y) (x * y' + y * x')
>   negate (Dual x x')        = Dual (negate x) (negate x')
>   signum _                  = undefined
>   abs _                     = undefined
> 
> instance Fractional Dual where
>   fromRational p = constD $ fromRational p
>   recip (Dual x x') = Dual (1.0 / x) (-x' / (x * x))

> instance Floating Dual where
>   pi = constD pi
>   exp   (Dual x x') = Dual (exp x)   (x' * exp x)
>   log   (Dual x x') = Dual (log x)   (x' / x)
>   sqrt  (Dual x x') = Dual (sqrt x)  (x' / (2 * sqrt x))
>   sin   (Dual x x') = Dual (sin x)   (x' * cos x)
>   cos   (Dual x x') = Dual (cos x)   (x' * (- sin x))
>   sinh  (Dual x x') = Dual (sinh x)  (x' * cosh x)
>   cosh  (Dual x x') = Dual (cosh x)  (x' * sinh x)
>   asin  (Dual x x') = Dual (asin x)  (x' / sqrt (1 - x*x))
>   acos  (Dual x x') = Dual (acos x)  (x' / (-sqrt (1 - x*x)))
>   atan  (Dual x x') = Dual (atan x)  (x' / (1 + x*x))
>   asinh (Dual x x') = Dual (asinh x) (x' / sqrt (1 + x*x))
>   acosh (Dual x x') = Dual (acosh x) (x' / (sqrt (x*x - 1)))
>   atanh (Dual x x') = Dual (atanh x) (x' / (1 - x*x))

Let us implement the function we considered earlier.

> f =  sqrt . (* 3) . sin

The compiler can infer its type

    [ghci]
    :t f

In [Haskell 98] only instances for *Float* and *Double* are defined
for the class *Floating*; we have made *Dual* an instance which means
we can now operate on values of this type as though, in some sense, they
are the same as *Float* or *Double*.

We know the derivative of the function and can also implement it
directly in Haskell.

> f' x = 3 * cos x / (2 * sqrt (3 * sin x)) 

Now we can evaluate the function along with its automatically
calculated derivative and compare that with the derivative we
calculated symbolically by hand.

    [ghci]
    f $ idD 2
    f' 2

To see that we are *not* doing symbolic differentiation (it's easy to
see we are not doing numerical differentiation) let us follow step
through the actual evaluation.

A Simple Application
====================

In order not to make this blog post too long let us apply AD to
finding parameters for a simple regression. The application to ANNs is
described in a previous [blog
post](http://idontgetoutmuch.wordpress.com/2013/05/31/neural-networks-and-automated-differentiation-3/).

Concluding Thoughts
===================

Efficiency
----------

Perhaps AD is underused because of efficiency?

It seems that the Financial Services industry is aware that AD is
*more* efficient than current practice albeit the technique is only
slowly permeating. Order of magnitude improvements have been reported.

* [Smoking Adjoints: fast evaluation of Greeks in Monte Carlo Calculations](http://www.risk.net/risk-magazine/technical-paper/1500261/smoking-adjoints-fast-monte-carlo-greeks)

* [Adjoints and automatic (algorithmic) differentiation in computational finance](http://arxiv.org/pdf/1107.1831.pdf)

Perhaps AD is slowly permeating into Machine Learning as well but
there seem to be no easy to find benchmarks.

* [Good Morning, Economics Blog](http://goodmorningeconomics.wordpress.com/2010/11/16/the-promise-of-bayesian-statistics-pt-2)

* [Andrew Gelman's Blog](http://andrewgelman.com/2013/03/04/stan-in-l-a-this-wed-330pm/)

Automatic Differentiation Tools
-------------------------------

If it were only possible to implement automatic differentiation in
Haskell then its applicability would be somewhat limited. Fortunately
this is not the case and it can be used in many languages.

In general, there are three different approaches:

* Operator overloading: available for Haskell and C++. See the Haskell
[ad package](http://hackage.haskell.org/package/ad) and the C++
[FADBAD](http://www.fadbad.com/fadbad.html) approach using templates.

* Source to source translators: available for Fortran and C.

* New languages with built-in AD primitives.


