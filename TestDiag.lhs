% Backpropogation is Just Steepest Descent with Automatic Differentiation
% Dominic Steinitz
% 20th September 2013

Introduction
============

This article is divided into

 * Refresher on neural networks

 * Methods for differentiation

 * Backpropogation

 * Concluding thoughts

Neural Network Refresher
========================

Here is our model, with $\boldsymbol{x}$ the input,
$\hat{\boldsymbol{y}}$ the predicted output and $\boldsymbol{y}$ the
actual output and $w^{(k)}$ the weights in the $k$-th layer.

$$
\begin{aligned}
a_j^{(1)} &= \sum_{i=0}^{M^{(1)}} w_{ij}^{(1)} x_i \\
z_j^{(1)} &= \tanh(a_j^{(1)}) \\
a_j^{(2)} &= \sum_{i=0}^{M^{(2)}} w_{ij}^{(2)} z_i^{(1)} \\
\dots     &= \ldots \\
a_j^{(L-1)} &= \sum_{i=0}^{M^{(L-1)}} w_{ij}^{(L-1)} z_i^{(L-2)} \\
z_j^{(L-1)} &= \tanh(a_j^{(L-1)}) \\
\hat{y}_j &= \sum_{i=0}^{M^{(L)}} w_{ij}^{(L)} z_i^{(L-1)} \\
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