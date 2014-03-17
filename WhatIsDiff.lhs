This is a test.

$$
\begin{aligned}
u_7 &= u_6 + u_ 5 \\
u_6 &= \exp{u_4} \\
u_5 &= \sin{u_4} \\
u_4 &= u_3 + u_2 \\
u_3 &= u_2^2 \\
u_2 &= \exp{u_1}
\end{aligned}
$$

$$
\mathbb{R}   \overset{f_1}{\longrightarrow}
\mathbb{R}   \overset{\langle f_3, f_2\rangle}{\longrightarrow}
\mathbb{R}^2 \overset{f_4}{\longrightarrow}
\mathbb{R}   \overset{\langle f_6, f_5\rangle}{\longrightarrow}
\mathbb{R}^2 \overset{f_7}{\longrightarrow}
\mathbb{R}
$$

where

$$
\begin{aligned}
f_7(x,y) &= x + y \\
f_6(x) &= \exp{x} \\
f_5(x) &= \sin{x} \\
f_4(x,y) &= x + y \\
f_3(x) &= x^2 \\
f_2(x) &= x \\
f_1(x) &= \exp{x}
\end{aligned}
$$

Thus $Df_7(u_6,u_5)$ is a linear map from the tangent space of
$\mathbb{R}^2$ at $(u_6,u_5)$ to the tangent space of $\mathbb{R}$ at
$f_7(u_6,u_5)$. We can represent this by a $1 \times 2$ matrix:

$$
\left( \begin{array}{cc}
\frac{\partial f_7}{\partial x}\bigg|_{x=u_6} & \frac{\partial f_7}{\partial y}\bigg|_{y=u_5} \end{array} \right)
=
\left( \begin{array}{cc}
1.0 & 1.0 \end{array} \right)
$$

Applying the chain rule to $D(f_7 \circ \langle f_6, f_5\rangle)$ we
get $Df_7(f_6(u_4), f_5(u_4)) \circ D\langle f_6, f_5\rangle(u_4)$. We can
represent $D\langle f_6, f_5\rangle(u_4)$ as a $2 \times 1$ matrix:

$$
\left( \begin{array}{c}
\frac{\partial f_6}{\partial x}\bigg|_{x=u_4} \\
\frac{\partial f_5}{\partial x}\bigg|_{x=u_4} \end{array} \right)
=
\left( \begin{array}{c}
\exp{u_4} \\
\cos{u_4} \end{array} \right)
$$

We can therefore write $D(f_7 \circ \langle f_6, f_5\rangle)\big|_{u_4}$ as

$$
\left( \begin{array}{cc}
\frac{\partial f_7}{\partial x}\bigg|_{x=f_6(u_4)} & \frac{\partial f_7}{\partial y}\bigg|_{y=f_5(u_4)} \end{array} \right)
\left( \begin{array}{c}
\frac{\partial f_6}{\partial x}\bigg|_{x=u_4} \\
\frac{\partial f_5}{\partial x}\bigg|_{x=u_4} \end{array} \right)
=
\frac{\partial f_7}{\partial u_6}\frac{\partial f_6}{\partial x} +
\frac{\partial f_7}{\partial u_5}\frac{\partial f_5}{\partial x}
=
\left( \begin{array}{cc}
1.0 & 1.0 \end{array} \right)
\left( \begin{array}{c}
\exp{u_4} \\
\cos{u_4} \end{array} \right)
=
\exp{u_4} + \cos{u_4}
$$

remembering that the partial differentials of $f_7$ are evalutated at
$f_6(u_4)$ and $f_5(u_4)$ respectively and the differentials of $f_6$ and
$f_5$ are evaluated at $u_4$.

Applying the chain rule to $D(f_7 \circ \langle f_6, f_5\rangle \circ
f_4)$ we get

$$
Df_7(f_6(f_4(u_3,u_2)), f_5(f_4(u_3,u_2))) \circ D\langle f_6,
f_5\rangle(f_4(u_3,u_2)) \circ Df_4(u_3,u_2)
$$.

We can represent $Df_4(u_3,u_2)$ by a $1 \times 2$ matrix:

$$
\left( \begin{array}{cc}
\frac{\partial f_4}{\partial x}\bigg|_{x=u_3} & \frac{\partial f_4}{\partial y}\bigg|_{y=u_2} \end{array} \right)
$$

The notation is starting to get a bit cluttered so let's drop that bit
of it which indicates at what value we evaluate the derivatives
bearing in mind that they need to be evaluated at the correct values.
We can therefor write $D(f_7 \circ \langle f_6, f_5\rangle \circ
f_4)\big|_(u_3,u_2)$ as

$$
\left( \begin{array}{cc}
\frac{\partial f_7}{\partial x}\bigg|_{x=f_6(f_4(a))} & \frac{\partial f_7}{\partial y}\bigg|_{y=f_5(f_4(a))} \end{array} \right)
\left( \begin{array}{c}
\frac{\partial f_6}{\partial x}\bigg|_{x=f_4(a)} \\
\frac{\partial f_5}{\partial x}\bigg|_{x=f_4(a)} \end{array} \right)
\left( \begin{array}{cc}
\frac{\partial f_4}{\partial x}\bigg|_{x=a} & \frac{\partial f_4}{\partial y}\bigg|_{y=b} \end{array} \right)
=
\left( \begin{array}{cc}
\frac{\partial f_7}{\partial u_6}\frac{\partial f_6}{\partial u_4}\frac{\partial f_4}{\partial x} +
\frac{\partial f_7}{\partial u_5}\frac{\partial f_5}{\partial u_4}\frac{\partial f_4}{\partial x} &
\frac{\partial f_7}{\partial u_6}\frac{\partial f_6}{\partial u_4}\frac{\partial f_4}{\partial y} +
\frac{\partial f_7}{\partial u_5}\frac{\partial f_5}{\partial u_4}\frac{\partial f_4}{\partial y}
\end{array} \right)
$$
