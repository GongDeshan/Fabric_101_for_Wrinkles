# Diff Friction Formular

## General Formular
$$
\frac{\partial E}{\partial \tau} = \mu N(x^n) \frac{\partial f}{\partial \tau}
$$

$$
\frac{\partial E}{\partial x} = \mu N(x^n) \frac{\partial f}{\partial \tau} T(x^n)
$$

To be brief, we write $N(x^n)$ as $N^n$ and $T(x^n)$ as $T^n$.


$$
\tau = T^n dx
$$

$$
dx = x - x^n
$$

Using Einstein notation, we can write the above equations as:

$$
\frac{\partial E}{\partial x_j} = \mu N^n \frac{\partial f}{\partial \tau_i} T_{ij}
$$

$$
\frac{\partial^2 E}{\partial x_j \partial x^n_k} = 

\mu \frac{\partial N^n}{\partial x^n_k}\frac{\partial f}{\partial \tau_i} T_{ij} 

+ \mu N^n \frac{\partial^2 f}{\partial \tau_i \partial \tau_l} \frac{\partial \tau_l}{\partial x^n_k}  T_{ij} 

+ \mu N^n \frac{\partial f}{\partial \tau_i} \frac{\partial T_{ij}}{\partial x^n_k}
$$

$$
\frac{\partial^2 E}{\partial x_j \partial x^n_k} = 

\mu \frac{\partial N^n}{\partial x^n_k}\frac{\partial f}{\partial \tau_i} T_{ij} 

+ \mu N^n \frac{\partial^2 f}{\partial \tau_i \partial \tau_l} \left(\frac{\partial T_{lp}}{\partial x^n_k} dx_p-T_{lk} \right)  T_{ij} 

+ \mu N^n \frac{\partial f}{\partial \tau_i} \frac{\partial T_{ij}}{\partial x^n_k}
$$

using Matrix notation, we can write the above equation as:

$$
\frac{\partial^2 E}{\partial x_j \partial x^n_k} = 

\mu \left(T^T \frac{\partial f}{\partial \tau} \right) \left(\frac{\partial N^n}{\partial D}\frac{\partial D}{\partial x^n} \right)^T 

+ \mu N^n\left(T^T\frac{\partial^2 f}{\partial \tau^2} \left( A - T\right)  + B\right)
$$

where:

$$
A_{lk} = \frac{\partial T_{lp}}{\partial x^n_k} dx_p
$$

$A = A_{2 \times 3N}$ and $B = B_{3N \times 3N}$. $N$ is the number of contact vertices.

$$
B_{jk} =\frac{\partial f}{\partial \tau_i} \frac{\partial T_{ij}}{\partial x^n_k} 
$$

So what we need to do first is calculate $\frac{\partial T_{ij}}{\partial x^n_k}$ for different primitive pairs, e.g. Point-Triangle, Edge-Edge, Point-Edge, Point-Point, etc.

I prefer to use `sympy` to generate the $\frac{\partial T_{ij}}{\partial x^n_k}$.

So for each primitive pair, we need to provide:

1. $\frac{\partial D}{\partial x^n_k}$
2. $A_{2 \times 3N}$
3. $B_{3N \times 3N}$
4. $T_{2 \times 3N}$
## Normal Force

Here we define:
$$
D:=d^2
$$

The derivative of d with respect to D is:
$$
\frac{\partial d}{\partial D} = \frac{1}{2d}
$$

So:

$$
N = - \frac{\partial B}{\partial d} = - \frac{\partial (\kappa B)}{\partial D} \frac{\partial D}{\partial d} = - \frac{\partial (\kappa B)}{\partial D} 2d
$$

Cleaning up, we get:

$$
N = -2 \frac{\partial (\kappa B)}{\partial D} d
$$

considering the derivative of N with respect to x:

$$
\frac{\partial N}{\partial x} 
= \frac{\partial N}{\partial D} \frac{\partial D}{\partial x} 
$$

$D$ is the interface, when apply $\frac{\partial D}{\partial x}$, we can serve for any kind of primitive pairs.

So we just consider the derivative of $N$ with respect to $D$.

$$
\frac{\partial N}{\partial D} 
= -2 (\frac{\partial^2 (\kappa B)}{\partial D^2} d + \frac{\partial (\kappa B)}{\partial D} \frac{\partial d}{\partial D})
$$

## $T$ For Point-HalfPlane

$$
\tau = T dx 
= 
\begin{bmatrix}
\hat{e}_1^T \\
\hat{e}_2^T
\end{bmatrix}
\cdot dx
$$

$$
\frac{\partial T_{ij}}{\partial x^n} = 0
$$

So $A= 0$ and $B= 0$, the result equation is:

$$
\frac{\partial^2 E}{\partial x_j \partial x^n_k} = 

\mu \left(T^T \frac{\partial f}{\partial \tau} \right) \left(\frac{\partial N^n}{\partial D}\frac{\partial D}{\partial x^n} \right)^T 

+ \mu N^n\left(-T^T\frac{\partial^2 f}{\partial \tau^2} T\right)
$$

## $T$ For Point-Triangle

Say we have a point $\vec{P}$ and a triangle $\vec{T_0}, \vec{T_1}, \vec{T_2}$.

Define the edge vectors $e_1 := \vec{T_1} - \vec{T_0}$ and $e_2 := \vec{T_2} - \vec{T_0}$, $e_3 := \vec{P} - \vec{T_0}$.

Define the relative movement vector $dx := \vec{x} - \vec{x^n}$, which is concretely defined as:

$$
dx =
\begin{bmatrix}
\vec{P} - \vec{P^n} \\
\vec{T_0} - \vec{T_0^n} \\
\vec{T_1} - \vec{T_1^n} \\
\vec{T_2} - \vec{T_2^n}
\end{bmatrix}
$$

$$
dr = 
\begin{bmatrix}
1 &   &   & -\beta_ 0 &   &   & -\beta_1 &   &   & -\beta_2 &   &   \\
  & 1 &   &   & -\beta_0  &   &   & -\beta_1 &   &   & -\beta_2 &   \\
  &   & 1 &   &   & -\beta_0  &   &   & -\beta_1 &   &   & -\beta_2
\end{bmatrix}
dx
$$

where,

$$
\beta_0 = 1 - \beta_1 - \beta_2
$$

$$
\beta_1 = D_1 / D
$$

$$
\beta_2 = D_2 / D
$$

$$
D = e_1^2 e_2^2 - (e_1 \cdot e_2)^2
$$

$$
D_1 = (e_1 \cdot e_3) e_2^2 - (e_1 \cdot e_2) (e_2 \cdot e_3)
$$

$$
D_2 = e_1^2 (e_2 \cdot e_3) - (e_1 \cdot e_3) (e_1 \cdot e_2)
$$

Find tangent space:

$$
\hat{e_1} = \frac{e_1}{\|e_1\|}
$$

$$
\hat{e_2} = \frac{e_1 \times (e_1 \times e_2)}{\|e_1 \times (e_1 \times e_2)\|}
$$

So finally, we have T:
$$
T_{2 \times 12} =
\begin{bmatrix}
\hat{e_1}^T & -\beta_0 \hat{e_1}^T & -\beta_1 \hat{e_1}^T & -\beta_2 \hat{e_1}^T \\
\hat{e_2}^T & -\beta_0 \hat{e_2}^T & -\beta_1 \hat{e_2}^T & -\beta_2 \hat{e_2}^T
\end{bmatrix}
$$

## $T$ For Point-Point

Say we have two points $\vec{P_0}$ and $\vec{P_1}$.

We define $n = \vec{P_1} - \vec{P_0}$

The criterion $s = (n \times \hat{e_x})^2 - (n \times \hat{e_y})^2$, where $\hat{e_x}:=[1, 0, 0]^T$ and $\hat{e_y}:=[0, 1, 0]^T$.

$$
\hat{e_1} =
\left\{
\begin{aligned}
\frac{n \times \hat{e_x}}{\|n \times \hat{e_x}\|}, & \text{if } s \ge 0 \\
\frac{n \times \hat{e_y}}{\|n \times \hat{e_y}\|}, & \text{if } s < 0
\end{aligned}
\right.
$$

$$
\hat{e_2} = \frac{n \times \hat{e_1}}{\|n \times \hat{e_1}\|}
$$

$$
T_{2 \times 6} = 
\begin{bmatrix}
\hat{e}_1^T & -\hat{e}_1^T \\
\hat{e}_2^T & -\hat{e}_2^T
\end{bmatrix}
$$

## $T$ For Point-Edge

Say we have a point $\vec{P}$ and an edge $\vec{E_0}, \vec{E_1}$.

Define the edge vector $e := \vec{E_1} - \vec{E_0}$, $r := \vec{P} - \vec{E_0}$.

The tangential space is:

$$
\hat{e_1} = \frac{e}{\|e\|}
$$

$$
\hat{e_2} = \frac{e \times r}{\|e \times r\|}
$$

$$
\beta = \frac{e \cdot r}{e \cdot e}
$$

$$
\beta_0 = 1 - \beta
$$

$$
\beta_1 = \beta
$$

$$
T = 
\begin{bmatrix}
\hat{e_1}^T & -\beta_0\hat{e_1}^T & -\beta_1\hat{e_1}^T \\
\hat{e_2}^T & -\beta_0\hat{e_2}^T & -\beta_1\hat{e_2}^T
\end{bmatrix}
$$

## $T$ For Edge-Edge

Say we have two edges $\vec{E_0}, \vec{E_1}$ and $\vec{E_2}, \vec{E_3}$.

Define the edge vectors $e_a := \vec{E_1} - \vec{E_0}$, 
$e_b := \vec{E_3} - \vec{E_2}$, $n = e_a \times e_b$, $r = \vec{E_2} - \vec{E_0}$.

$$
\hat{e_1} = \frac{e_a}{\|e_a\|}
$$

$$
\hat{e_2} = \frac{n \times \hat{e_1}}{\|n \times \hat{e_1}\|}
$$

$$
\beta_0 = \frac{D_1}{D}
$$

$$
\beta_1 = \frac{D_2}{D}
$$

$$
D = e_a^2 e_b^2 - (e_a \cdot e_b)^2
$$

$$
D_1 = (r \cdot e_a) e_b^2 - (e_b \cdot e_a) (r \cdot e_b)
$$

$$
D_2 = - e_a^2 (r \cdot e_b) + (r \cdot e_a) (e_b \cdot e_a)
$$

$$
T = \begin{bmatrix}
  (1-\beta_0)\hat{e_1}^T & \beta_0\hat{e_1}^T & -(1-\beta_1)\hat{e_1}^T & -\beta_1\hat{e_1}^T \\
  (1-\beta_0)\hat{e_2}^T & \beta_0\hat{e_2}^T & -(1-\beta_1)\hat{e_2}^T & -\beta_1{e_2}^T
\end{bmatrix}
$$

## Diff Contact Coeff

### Frictional Contact

$$
G = \frac{\partial E}{\partial x} = \frac{\partial E}{\partial \tau} \frac{\partial \tau}{\partial x} = \mu N^n \frac{\partial f}{\partial \tau} T^n
$$

$$
\frac{\partial G}{\partial \mu} = N^n \frac{\partial f}{\partial \tau} T^n
$$

$$
\frac{\partial G}{\partial \kappa} = \mu \frac{\partial N^n}{\partial \kappa} \frac{\partial f}{\partial \tau} T^n
$$

### Normal Contact

$$
G = \frac{\partial (\kappa B)}{\partial x} = \kappa \frac{\partial B}{\partial x}
$$

$$
\frac{\partial G}{\partial \kappa} = \frac{\partial B}{\partial x}
$$
