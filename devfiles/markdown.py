[[BiSE Binarization]]

# Binary Dilation

The projected space is:

![[Conv for morphology#Prop 1.1 [Best Inequality]]

Let $A(S)$ be the convex activated space defined as
$$A(S) \defeq \Big\{(W, b) \in \R^{\Omega} \times \R ~|~\sum_{i \in \Omega \setminus S}w_k < b < \min_{i \in S}w_i  \Big\} = \bigg(\bigcap_{k \in S}\big\{(W, b) \in \R^{\Omega} \times \R ~|~ w_k>b\big\}\bigg) \bigcap \bigg(\big\{(W, b) \in \R^{\Omega} \times \R ~|~ \sum_{i \in \Omega \setminus S}w_i < b\big\}\bigg)
$$
Let $\widehat{W}, \hat{b}$ be a set of weights we want to project on $A(S)$. The problem is:

$$
\begin{align}
L(W, b) &\defeq \frac{1}{2} \sum_{k \in \Omega}(w_k - \hat{w}_k)^2 + \frac{1}{2}(b - \hat{b})^2 \\
(\text{PB}~1)~~\min_{W, b \in A(S)} &L(W,b)
\end{align}
$$

It can be rewritten as:
$$
\begin{align}
(\text{PB}~2)~~\min_{W, b \in \R^{\Omega}\times \R} L(W, b) \\
\text{s.t.}
\begin{cases}
\sum_{i \in \Omega \setminus S}w_i \leq b \\
\forall k \in S , -w_k \leq -b
\end{cases}
\end{align}
$$

The lagrangian is
$$
\mathcal{L}(W, b, \lambda) = L(W, b) + \lambda_0\Big(\sum_{i \in \Omega \setminus S}w_i - b\Big) + \sum_{k \in S}{\lambda_k(b - w_k)}
$$
We suppose that $A(S) \ne \emptyset$. Then, the Slater conditions are respected. Therefore, we can cancel the gradient of the lagrangien function.
$$
\begin{align}
\forall j \in S, \frac{\partial \mathcal{L}}{\partial w_j} = w_j - \hat{w}_j - \lambda_j = 0 \\
\forall i \in \Omega \setminus S, \frac{\partial \mathcal{L}}{\partial w_i} = w_i - \hat{w}_i + \lambda_0 = 0 \\
\frac{\partial \mathcal{L}}{\partial b} = b - \hat{b} - \lambda_0 +\sum_{j \in S}{\lambda_j} = 0 \\
\lambda_0\cdot\Big(\sum_{i \in \Omega \setminus S} - b\Big) = 0 \\
\forall j \in S, \lambda_j\cdot(b - w_j) = 0
\end{align}
$$