# Qubit state estimation

Quantum state estimation is an important task of many quantum information protocols. Here we consider the problem of complete qubit state estimation restricting ourselves to not performing measurements over the qubit of interest (would be te case of cavity QED). This can be achieved using two auxiliary qubits (meters) and performing measurements over them.

A Hamiltonian $\mathcal{H}$ describes the interaction between the meters $A$ and $B$, and the system of interest $S$; this hamiltonian depends on a set of parameters $\mathcal{H} = \mathcal{H}(\tilde{\Theta})$. Then we can build the evolution operator $\mathcal{U}=\exp{-iH/\hbar}$; which has a dependence on a (conveniently) redefined set of parameters $\mathcal{U} = \mathcal{U}(\Theta)$.
The complete initial system $\ket{\Psi_0} = \ket{+}^A\ket{\psi_0}\ket{+}^B$ evolves as $\ket{\Psi(\Theta)} = \mathcal{U}(\Theta)\ket{\Psi_0}$.
	
The meters $A$ and $B$ are measured in the $\sigma_x$ basis. The possible results with probabilities $p_{kl}$ ($k$ for $A$ and $l$ for $B$) with $k,l\in\{0,1\}$ ($\tilde{k}=(1-2k)$ and $\tilde{l}=(1-2l)$)  can be related with the bloch vector of the system $\vec{s} = (s_1,s_2,s_3)$ by the equation
$$\begin{equation*}
    p_{kl}	= \frac{1}{4}s_0+\Sigma_{\mu = 1}^{3}\left(a_\mu \tilde{k} + b_\mu \tilde{l} + c_\mu \tilde{k}\tilde{l}\right) s_\mu,\quad \text{with }s_0 = 1,
\end{equation*}$$
Here the coefficients $a_\mu, b_\mu$ and $c_\mu$ depend on the parameters $\Theta$ and are related to de POVM elements of the measurement.
We define a vector $\mathbf{p} = (p_{00},p_{01},p_{10},p_{11})^T$, the vector $\textbf{s} = (s_0,s_1,s_2,s_3)^T$, and the matrix $T$ with components $T_{kl} = \frac{1}{4}+\sum_{\mu = 0}^{3}\left(a_\mu k + b_\mu l + c_\mu kl\right)$ to write }
$$\begin{equation*}
    \textbf{s} = \mathbf{T}^{-1}\textbf{p}, 
\end{equation*}$$
that allows the estimation of the Bloch vector from the measurements.

This code implements this estimation technique and is tested in real quantum computers. You can see some results related to this work in this [pre print](https://arxiv.org/abs/2301.11121).