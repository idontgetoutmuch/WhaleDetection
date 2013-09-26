% Backpropogation is Just Steepest Descent with Automatic Differentiation
% Dominic Steinitz
% 20th September 2013

Introduction
============

The problem is simple to state: we have a (highly) non-linear
function, the cost function of an Artifical Neural Network (ANN), and
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
floating point arithmentic.

As an alternative we could turn our algorithm or computer program into
a function more recognisable as a mathematical function and then
compute the differential itself as a function either by hand or by
using a symbolic differentiation package. For the complicated
expression which is our mathematical function, the former would be
error prone and the latter could easily generate something which would
be even more complex and costly to evaluate than the original
expression.

The standard approach is to use a technique called backpropagation and
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

Neural Network Refresher
========================

Here is our model, with $\boldsymbol{x}$ the input,
$\hat{\boldsymbol{y}}$ the predicted output and $\boldsymbol{y}$ the
actual output and $w^{(k)}$ the weights in the $k$-th layer. We 

$$
\begin{aligned}
a_j^{(1)} &= \sum_{i=0}^{N^{(1)}} w_{ij}^{(1)} x_i \\
z_j^{(1)} &= \tanh(a_j^{(1)}) \\
a_j^{(2)} &= \sum_{i=0}^{N^{(2)}} w_{ij}^{(2)} z_i^{(1)} \\
\dots     &= \ldots \\
a_j^{(L-1)} &= \sum_{i=0}^{N^{(L-1)}} w_{ij}^{(L-1)} z_i^{(L-2)} \\
z_j^{(L-1)} &= \tanh(a_j^{(L-1)}) \\
\hat{y}_j &= \sum_{i=0}^{N^{(L)}} w_{ij}^{(L)} z_i^{(L-1)} \\
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
\Delta w_{ij} = -\frac{\partial E}{\partial w_{ij}}
$$

Applying the chain rule

$$
\Delta w_{ij} = -\frac{\partial E}{\partial w_{ij}} = -\frac{\partial E}{\partial a_i}\frac{\partial a_i}{\partial w_{ij}}
$$

Since

$$
a_j = \sum_{i=0}^M w_{ij}z_i
$$

we have

$$
\frac{\partial a_i}{\partial w_{ij}} = \frac{\sum_{l=0}^M w_{lj}z_l}{\partial w_{ij}} = z_i
$$

Defining

$$
\delta_j \equiv -\frac{\partial E}{\partial a_j}
$$

we obtain

$$
\Delta w_{ij} = -\frac{\partial E}{\partial w_{ij}} = \delta_j z_i
$$

Finding the $z_i$ for each layer is straightforward: we start with the
inputs and propagate forward. In order to find the $\delta_j$ we need
to start with the outputs a propagate backwards:

* For the output layer we have (since $\hat{y}_j = a_j$)

$$
\delta_j = \frac{\partial E}{\partial a_j} = \frac{\partial E}{\partial y_j} = \frac{\partial}{\partial y_j}\bigg(\frac{1}{2}\sum_{i=0}^M (\hat{y}_i - y_i)^2\bigg) = \hat{y}_j - y_j
$$



Differentiation
===============

We consider multi-layer perceptrons and use the term neural network
interchangeably. In summary we have a parameterised non-linear model
(the neural network), a cost function and some training data and we
wish to estimate the parameters in the neural network from the
training data so as to minimize the total cost

$$
E(\boldsymbol{w}; \boldsymbol{x}, \boldsymbol{y}) = \frac{1}{2}\|(\hat{\boldsymbol{y}} - \boldsymbol{y})\|^2
$$

where $\boldsymbol{w}$ is the vector of parameters in the model,
$\boldsymbol{x}$ are the inputs, $\boldsymbol{y}$ are the outputs and
$\hat{\boldsymbol{y}}$ are outputs predicted by our model. For now the
exact form of the model is not important.

In order to find the parameters that minimize the loss function we use
steepest (aka gradient) descent: we calculate the derivative of the
loss function and then step a small way in the direction that reduces
the cost the most. The fact that we are minimizing non-linear function
means that we may end up at a local minimum but that is just a fact of
life and we do not consider this here any further. So all we need is
the derivative of our cost function.

Let us consider some techniques for doing this.

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

Typically the non-linear function that a neural network gives is much
more complex than the simple function given above. Thus its derivative
will correspondingly more complex and therefore expensive to
compute. Moreover calculating this derivative by hand could easily
introduce errors. And in order to have a computer perform the symbolic
calculation we would have to encode our cost function somehow so that
it is amenable to this form of manipulation.

Numerical Differentiation
-------------------------

```{.dia width='500'}
import MLTalkDiagrams
dia = errDiag
```

Other
-----

Automatic differentiation is *not*

 * Symbolic differentiation or

 * Numerical differentiation

Consider the function

$$
f(x) = \exp(\exp(x) + (\exp(x))^2) + \sin(\exp(x) + (\exp(x))^2)
$$

Let us write this a data flow graph.

```{.dia width='500'}
import MLTalkDiagrams
dia = example
```

$$
\begin{aligned}
\frac{\mathrm{d}u_7}{\mathrm{d}u_7} &= 1 \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_6} &= 1 \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_5} &= 1 \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_4} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_6}\frac{\mathrm{d}u_6}{\mathrm{d}u_4} +
 \frac{\mathrm{d}u_7}{\mathrm{d}u_5}\frac{\mathrm{d}u_5}{\mathrm{d}u_4} \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_3} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_4}\frac{\mathrm{d}u_4}{\mathrm{d}u_3} \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_2} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_2}\frac{\mathrm{d}u_2}{\mathrm{d}u_4} +
 \frac{\mathrm{d}u_7}{\mathrm{d}u_3}\frac{\mathrm{d}u_3}{\mathrm{d}u_4} \\
\frac{\mathrm{d}u_7}{\mathrm{d}u_1} &=
 \frac{\mathrm{d}u_7}{\mathrm{d}u_2}\frac{\mathrm{d}u_2}{\mathrm{d}u_1}
\end{aligned}
$$


> data Dx = D Double Double
>         deriving (Eq, Show)
> 
> instance Num Dx where
>   (D x a) + (D y b) = D (x + y) (a + b)
>   