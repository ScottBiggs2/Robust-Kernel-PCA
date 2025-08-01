import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import svd, sqrtm
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
import warnings
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt

class RKPCA:
    """
    Robust Kernel Principal Component Analysis (RKPCA)
    
    Implementation based on:
    "Exactly Robust Kernel Principal Component Analysis" by Jicong Fan and Tommy W.S. Chow
    
    RKPCA decomposes a matrix M into X + E where:
    - X is the clean data matrix (potentially high-rank with low latent dimensionality)
    - E is the sparse noise matrix
    
    The algorithm minimizes: Tr(K^(p/2)) + λ||E||_1 subject to X + E = M
    where K is the kernel matrix of X and p is typically 1.
    
    For data compression, principal components are extracted from the kernel space.
    """
    
    def __init__(self, 
                 n_components=None,
                 kernel='rbf', 
                 sigma=None, 
                 beta=1.0,
                 lambda_0=0.5, 
                 p=1.0,
                 max_iter=100, 
                 tol=1e-6,
                 algorithm='plm_adss',
                 omega=0.1,
                 c=2.0,
                 verbose=False,
                 device='cpu'):
        """
        Initialize RKPCA
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep for dimensionality reduction.
            If None, keep all components
        kernel : str, default='rbf'
            Kernel type ('rbf', 'linear', 'polynomial')
        sigma : float, optional
            RBF kernel bandwidth parameter. If None, estimated from data
        beta : float, default=1.0
            Factor for automatic sigma estimation
        lambda_0 : float, default=0.5
            Regularization parameter for sparse term
        p : float, default=1.0
            Schatten p-norm parameter (1.0 for nuclear norm)
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-6
            Convergence tolerance
        algorithm : str, default='plm_adss'
            Optimization algorithm ('plm_adss' or 'admm_btls')
        omega : float, default=0.1
            Initial step size parameter for PLM+AdSS
        c : float, default=2.0
            Step size increase factor
        verbose : bool, default=False
            Print iteration information
        device : str, default='cpu'
            Computing device ('cpu' or 'cuda')
        """
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.beta = beta  
        self.lambda_0 = lambda_0
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.omega = omega
        self.c = c
        self.verbose = verbose
        self.device = device
        
        # Results
        self.X_ = None
        self.E_ = None
        self.K_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None
        self.X_transformed_ = None
        self.objective_values_ = []
        
    def _estimate_sigma(self, M: np.ndarray) -> float:
        """
        Estimate RBF kernel bandwidth parameter following MATLAB implementation
        MATLAB: XX=sum(X.*X,1); D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
                ker.par=(ker.c*mean(real(D(:).^0.5)))^2; (where ker.c defaults to 1)
        """
        n = M.shape[1]
        if n > 1000:  # Sample for large datasets
            idx = np.random.choice(n, 1000, replace=False)
            M_sample = M[:, idx]
        else:
            M_sample = M
            
        # Compute pairwise squared distances exactly like MATLAB
        XX = np.sum(M_sample * M_sample, axis=0)  # sum along features
        D = XX[:, None] + XX[None, :] - 2 * M_sample.T @ M_sample
        
        # Take square root and mean, then square (following MATLAB logic)
        mean_dist = np.mean(np.sqrt(np.maximum(D, 0)))  # Ensure non-negative for sqrt
        sigma = self.beta * mean_dist
        
        return sigma
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K"""
        if self.kernel == 'rbf':
            if self.sigma is None:
                self.sigma = self._estimate_sigma(X)
            # Using sklearn's rbf_kernel which handles the gamma parameter
            gamma = 1.0 / (2 * self.sigma ** 2)
            K = rbf_kernel(X.T, gamma=gamma)
        elif self.kernel == 'linear':
            K = X.T @ X
        elif self.kernel == 'polynomial':
            K = pairwise_kernels(X.T, metric='polynomial', degree=2)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        
        return K
    
    def _extract_principal_components(self, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract principal components from kernel matrix
        
        Parameters:
        -----------
        K : np.ndarray
            Kernel matrix
            
        Returns:
        --------
        eigenvalues : np.ndarray
            Eigenvalues in descending order
        eigenvectors : np.ndarray  
            Eigenvectors corresponding to eigenvalues
        """
        # Center the kernel matrix (important for PCA)
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep only positive eigenvalues (numerical stability)
        positive_idx = eigenvalues > 1e-12
        eigenvalues = eigenvalues[positive_idx]
        eigenvectors = eigenvectors[:, positive_idx]
        
        return eigenvalues, eigenvectors
    
    def _project_to_feature_space(self, eigenvectors: np.ndarray, eigenvalues: np.ndarray, 
                                 n_components: Optional[int] = None) -> np.ndarray:
        """
        Project data to principal component space
        
        Parameters:
        -----------
        eigenvectors : np.ndarray
            Eigenvectors from kernel matrix
        eigenvalues : np.ndarray
            Eigenvalues from kernel matrix
        n_components : int, optional
            Number of components to keep
            
        Returns:
        --------
        X_transformed : np.ndarray, shape (n_components, n_samples)
            Transformed data in principal component space
        """
        if n_components is None:
            n_components = len(eigenvalues)
        else:
            n_components = min(n_components, len(eigenvalues))
        
        # Select top n_components
        eigenvalues_selected = eigenvalues[:n_components]
        eigenvectors_selected = eigenvectors[:, :n_components]
        
        # Transform: multiply by sqrt of eigenvalues for proper scaling
        X_transformed = (eigenvectors_selected * np.sqrt(eigenvalues_selected)).T
        
        return X_transformed
    
    def _compute_kernel_matrix_matlab_style(self, X: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix following the MATLAB implementation exactly
        """
        if self.kernel == 'rbf':
            if self.sigma is None:
                self.sigma = self._estimate_sigma(X)
            
            # Compute pairwise squared distances efficiently
            # MATLAB: XX=sum(X.*X,1); D=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
            XX = np.sum(X * X, axis=0)  # Shape: (n,)
            D = XX[:, None] + XX[None, :] - 2 * X.T @ X
            
            # RBF kernel: K = exp(-D/(2*ker.par)) where ker.par = sigma^2
            # So we use: K = exp(-D/(2*sigma^2))
            K = np.exp(-D / (2 * self.sigma ** 2))
            
        elif self.kernel == 'linear':
            K = X.T @ X
        elif self.kernel == 'polynomial':
            K = pairwise_kernels(X.T, metric='polynomial', degree=2)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        
        return K
    
    def _compute_gradient_matlab_style(self, X: np.ndarray, E: np.ndarray, 
                                     gLgK: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradient following MATLAB implementation exactly
        """
        d, n = X.shape
        
        # T = gLgK .* K (element-wise multiplication)
        T = gLgK * K
        
        # B = repmat(sum(T), d, 1)
        B = np.tile(np.sum(T, axis=0), (d, 1))
        
        # Gradient: g = -2/sigma * ((X-E)*T - (X-E).*B)
        X_minus_E = X - E
        g = -2.0 / self.sigma * (X_minus_E @ T - X_minus_E * B)
        
        # Lipschitz estimate: L = ||2/sigma * (T - I*mean(B))||
        mean_B = np.mean(B)
        L = np.linalg.norm(2.0 / self.sigma * (T - np.eye(n) * mean_B), ord=2)
        
    def _soft_threshold(self, X: np.ndarray, tau: float) -> np.ndarray:
        """Soft thresholding operator"""
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    
    def _compute_objective(self, K: np.ndarray, E: np.ndarray, lam: float) -> float:
        """Compute objective function value"""
        # Compute K^(p/2) using eigendecomposition for numerical stability
        eigenvals, eigenvecs = np.linalg.eigh(K)
        eigenvals = np.maximum(eigenvals, 1e-12)  # Avoid numerical issues
        
        if self.p == 1.0:
            obj_k = np.sum(np.sqrt(eigenvals))
        else:
            obj_k = np.sum(eigenvals ** (self.p / 2))
        
        obj_e = lam * np.sum(np.abs(E))
        return obj_k + obj_e
    
    def _compute_kernel_gradient(self, X: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute gradient of Tr(K^(p/2)) with respect to X"""
        n = X.shape[1]
        
        # Compute K^(p/2-1) using eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(K)
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        if self.p == 1.0:
            # For p=1, we need K^(-1/2)
            K_power = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        else:
            K_power = eigenvecs @ np.diag(eigenvals ** (self.p/2 - 1)) @ eigenvecs.T
        
        # For RBF kernel: dK_ij/dX = 2/(sigma^2) * (x_i - x_j) * K_ij
        if self.kernel == 'rbf':
            H = (self.p / 2) * K_power * K
            grad = np.zeros_like(X)
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = X[:, i] - X[:, j]
                        grad[:, i] += (2.0 / (self.sigma ** 2)) * H[i, j] * diff
                        
        elif self.kernel == 'linear':
            # For linear kernel: K = X^T X, so dK/dX = 2X * H
            H = (self.p / 2) * K_power
            grad = 2 * X @ H
        else:
            # Numerical gradient for other kernels
            grad = self._numerical_gradient(X, K)
            
        return grad
    
    def _numerical_gradient(self, X: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute numerical gradient (fallback for complex kernels)"""
        grad = np.zeros_like(X)
        eps = 1e-6
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i, j] += eps
                X_minus[i, j] -= eps
                
                K_plus = self._compute_kernel_matrix(X_plus)
                K_minus = self._compute_kernel_matrix(X_minus)
                
                obj_plus = self._compute_objective(K_plus, np.zeros_like(X), 0)
                obj_minus = self._compute_objective(K_minus, np.zeros_like(X), 0)
                
                grad[i, j] = (obj_plus - obj_minus) / (2 * eps)
                
        return grad
    
    def _plm_adss_step(self, M: np.ndarray, E: np.ndarray, lam: float, nu: float) -> np.ndarray:
        """One step of Proximal Linearized Minimization with Adaptive Step Size"""
        X = M - E
        K = self._compute_kernel_matrix(X)
        
        # Compute gradient
        grad_E = -self._compute_kernel_gradient(X, K)
        
        # Proximal step
        E_new = self._soft_threshold(E - grad_E / nu, lam / nu)
        
        return E_new
    
    def _fit_plm_adss(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit using Proximal Linearized Minimization with Adaptive Step Size (MATLAB style)"""
        d, n = M.shape
        
        # Initialize following MATLAB code
        E = np.zeros_like(M)
        lam = self.lambda_0 / np.sum(np.abs(M)) * n  # Normalize lambda
        c = self.omega  # Step size parameter
        
        if self.verbose:
            print(f'lambda={lam:.6f}')
        
        # Estimate sigma if needed
        if self.kernel == 'rbf' and self.sigma is None:
            self.sigma = self._estimate_sigma(M)
            if self.verbose:
                print(f'kernel type: {self.kernel}  kernel parameter: {self.sigma:.6f}')
        
        normF_X = np.linalg.norm(M, 'fro')
        
        for iteration in range(self.max_iter):
            # Compute kernel matrix K = ker_x(X-E, ker)
            X_current = M - E
            K = self._compute_kernel_matrix_matlab_style(X_current)
            
            # Compute K^(p/2) and gradient term gLgK
            eigenvals, eigenvecs = np.linalg.eigh(K)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Numerical stability
            
            # Kp2 = real(K^(p/2))
            Kp2_eigenvals = eigenvals ** (self.p / 2)
            Kp2 = eigenvecs @ np.diag(Kp2_eigenvals) @ eigenvecs.T
            
            # gLgK = real(p/2 * Kp2 * (K + eye(n)*1e-5)^(-1))
            K_reg = K + np.eye(n) * 1e-5
            try:
                K_inv = np.linalg.inv(K_reg)
                gLgK = (self.p / 2) * Kp2 @ K_inv
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                gLgK = (self.p / 2) * Kp2 @ np.linalg.pinv(K_reg)
            
            # Objective function
            f = np.trace(Kp2)
            obj_val = f + lam * np.sum(np.abs(E))
            self.objective_values_.append(obj_val)
            
            # Compute gradient following MATLAB gLgX_m function
            # The gradient is with respect to X, but we need it with respect to E
            if self.kernel == 'rbf':
                # T = gLgK .* K (element-wise multiplication)
                T = gLgK * K
                
                # B = repmat(sum(T), d, 1) - sum along columns, replicate along rows
                B = np.tile(np.sum(T, axis=0), (d, 1))
                
                # Gradient w.r.t. X: gX = -2/ker.par * ((X-E)*T - (X-E).*B)
                X_minus_E = X_current
                gX = -2.0 / (self.sigma ** 2) * (X_minus_E @ T - X_minus_E * B)
                
                # Since E = M - X, gradient w.r.t. E is -gradient w.r.t. X
                gE = -gX
                
                # Lipschitz estimate: L = ||2/ker.par * (T - I*mean(B))||
                mean_B = np.mean(B)
                L = np.linalg.norm(2.0 / (self.sigma ** 2) * (T - np.eye(n) * mean_B), ord=2)
            else:
                raise NotImplementedError("Only RBF kernel supported in PLM+AdSS")
            
            # Update step size
            tau = c * L
            
            # Soft thresholding: E_new = soft_threshold(E - gE/tau, lambda/tau)
            temp = E - gE / tau
            E_new = np.maximum(0, temp - lam/tau) + np.minimum(0, temp + lam/tau)
            
            # Adaptive step size
            if iteration > 0:
                if obj_val > self.objective_values_[-2]:
                    c = min(5, c * 1.2)
            
            # Check convergence
            stopC = np.linalg.norm(E_new - E, 'fro') / normF_X
            
            if self.verbose and (iteration % 100 == 0 or stopC < self.tol or iteration == 0):
                print(f'iteration={iteration+1}/{self.max_iter}  obj={obj_val:.6f}  '
                      f'stopC={stopC:.6f}  tau={tau:.6f}  c={c:.6f}')
            
            if stopC < self.tol and iteration > 50:
                if self.verbose:
                    print('converged')
                break
            
            E = E_new
        
        X = M - E
        return X, E
    
    def _fit_admm_btls(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit using ADMM with Backtracking Line Search"""
        d, n = M.shape
        
        # Initialize
        X = M.copy()
        E = np.zeros_like(M)
        Q = np.zeros_like(M)  # Lagrange multipliers
        
        lam = self.lambda_0 * n / np.sum(np.abs(M))
        mu = 10 * lam  # Penalty parameter
        
        for iteration in range(self.max_iter):
            # Update X with backtracking line search
            X_prev = X.copy()
            K = self._compute_kernel_matrix(X)
            grad_X = self._compute_kernel_gradient(X, K) + mu * (X + E - M + Q / mu)
            
            # Backtracking line search
            eta = 1.0
            while eta > 1e-10:
                X_new = X - eta * grad_X
                K_new = self._compute_kernel_matrix(X_new)
                
                # Check Armijo condition
                obj_new = self._compute_objective(K_new, E, lam) + \
                          mu/2 * np.sum((X_new + E - M + Q/mu)**2)
                obj_old = self._compute_objective(K, E, lam) + \
                          mu/2 * np.sum((X + E - M + Q/mu)**2)
                
                if obj_new <= obj_old - 0.1 * eta * np.sum(grad_X**2):
                    X = X_new
                    break
                eta *= 0.5
            
            # Update E (soft thresholding)
            E = self._soft_threshold(M - X - Q/mu, lam/mu)
            
            # Update Lagrange multipliers
            Q = Q + mu * (X + E - M)
            
            # Check convergence
            primal_residual = np.linalg.norm(X + E - M, 'fro')
            if primal_residual < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            if self.verbose and iteration % 10 == 0:
                current_obj = self._compute_objective(self._compute_kernel_matrix(X), E, lam)
                self.objective_values_.append(current_obj)
                print(f"Iteration {iteration}: Objective = {current_obj:.6f}, "
                      f"Residual = {primal_residual:.6f}")
        
        return X, E
    
    def fit(self, M: np.ndarray) -> 'RKPCA':
        """
        Fit RKPCA to data matrix M
        
        Parameters:
        -----------
        M : np.ndarray, shape (d, n)
            Data matrix where each column is a sample
            
        Returns:
        --------
        self : RKPCA
            Fitted estimator
        """
        M = np.asarray(M, dtype=np.float64)
        
        if M.ndim != 2:
            raise ValueError("M must be a 2D array")
        
        # Estimate sigma if not provided
        if self.kernel == 'rbf' and self.sigma is None:
            self.sigma = self._estimate_sigma(M)
            if self.verbose:
                print(f"Estimated sigma: {self.sigma:.4f}")
        
        # Choose optimization algorithm
        if self.algorithm == 'plm_adss':
            self.X_, self.E_ = self._fit_plm_adss(M)
        elif self.algorithm == 'admm_btls':
            self.X_, self.E_ = self._fit_admm_btls(M)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Store final kernel matrix and extract components
        self.K_ = self._compute_kernel_matrix_matlab_style(self.X_)
        
        # Extract principal components for dimensionality reduction
        self.eigenvalues_, self.eigenvectors_ = self._extract_principal_components(self.K_)
        
        # Project to reduced space if n_components is specified
        if self.n_components is not None:
            self.X_transformed_ = self._project_to_feature_space(
                self.eigenvectors_, self.eigenvalues_, self.n_components)
            
            if self.verbose:
                explained_var_ratio = np.sum(self.eigenvalues_[:self.n_components]) / np.sum(self.eigenvalues_)
                print(f"Explained variance ratio with {self.n_components} components: {explained_var_ratio:.4f}")
        
        return self
    
    def transform(self, X: Optional[np.ndarray] = None, n_components: Optional[int] = None) -> np.ndarray:
        """
        Transform data to reduced dimensionality or kernel space
        
        Parameters:
        -----------
        X : np.ndarray, optional
            Data to transform. If None, uses fitted data
        n_components : int, optional
            Number of components to keep. If None, uses self.n_components
            
        Returns:
        --------
        X_transformed : np.ndarray
            Transformed data. Shape depends on n_components:
            - If n_components specified: (n_components, n_samples)
            - If n_components is None: full kernel matrix (n_samples, n_samples)
        """
        if self.eigenvectors_ is None or self.eigenvalues_ is None:
            raise ValueError("Must fit model before transforming")
        
        if n_components is None:
            n_components = self.n_components
            
        if X is None:
            if n_components is not None:
                # Return pre-computed transformation
                if self.X_transformed_ is not None and self.X_transformed_.shape[0] == n_components:
                    return self.X_transformed_
                else:
                    # Recompute with different n_components
                    return self._project_to_feature_space(
                        self.eigenvectors_, self.eigenvalues_, n_components)
            else:
                # Return full kernel matrix
                return self.K_
        else:
            # Transform new data
            if X.shape[0] != self.X_.shape[0]:
                raise ValueError("X must have same number of features as training data")
            
            # Compute kernel matrix between new data and training data
            if self.kernel == 'rbf':
                # Compute K(X_new, X_train) using same sigma as training
                XX_new = np.sum(X * X, axis=0)
                XX_train = np.sum(self.X_ * self.X_, axis=0)
                D = XX_new[:, None] + XX_train[None, :] - 2 * X.T @ self.X_
                K_new = np.exp(-D / (2 * self.sigma ** 2))
            elif self.kernel == 'linear':
                K_new = X.T @ self.X_
            else:
                raise NotImplementedError("Transform not implemented for this kernel")
            
            # Center the kernel matrix
            n_train = self.X_.shape[1]
            n_new = X.shape[1]
            
            one_n_train = np.ones(n_train) / n_train
            one_n_new = np.ones(n_new) / n_new
            
            # Center K_new
            K_new_centered = (K_new - 
                            np.outer(one_n_new, np.sum(self.K_, axis=0) / n_train) -
                            np.outer(np.sum(K_new, axis=1) / n_train, one_n_train) +
                            np.sum(self.K_) / (n_train ** 2))
            
            if n_components is not None:
                # Project to principal component space
                eigenvecs_selected = self.eigenvectors_[:, :n_components]
                eigenvals_selected = self.eigenvalues_[:n_components]
                
                # Transform: K_new_centered @ eigenvectors / sqrt(eigenvalues)
                X_transformed = (K_new_centered @ eigenvecs_selected / np.sqrt(eigenvals_selected)).T
                return X_transformed
            else:
                return K_new_centered
    
    def fit_transform(self, M: np.ndarray, n_components: Optional[int] = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Fit RKPCA and return results
        
        Parameters:
        -----------
        M : np.ndarray, shape (d, n)
            Data matrix
        n_components : int, optional
            If specified, return compressed representation instead of decomposition
            
        Returns:
        --------
        If n_components is None:
            X : np.ndarray, shape (d, n) - Clean data matrix
            E : np.ndarray, shape (d, n) - Sparse error matrix
        If n_components is specified:
            X_compressed : np.ndarray, shape (n_components, n) - Compressed representation
        """
        self.fit(M)
        
        if n_components is not None:
            return self.transform(n_components=n_components)
        else:
            return self.X_, self.E_
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back from reduced space (approximate reconstruction)
        
        Parameters:
        -----------
        X_transformed : np.ndarray, shape (n_components, n_samples)
            Data in reduced space
            
        Returns:
        --------
        X_reconstructed : np.ndarray, shape (d, n_samples)
            Reconstructed data in original space
        """
        if self.eigenvectors_ is None or self.eigenvalues_ is None:
            raise ValueError("Must fit model before inverse transforming")
        
        n_components = X_transformed.shape[0]
        
        # Get corresponding eigenvectors and eigenvalues
        eigenvecs_selected = self.eigenvectors_[:, :n_components]
        eigenvals_selected = self.eigenvalues_[:n_components]
        
        # Reconstruct in kernel space: K_reconstructed = eigenvectors @ (X_transformed * sqrt(eigenvalues))
        # X_transformed is (n_components, n_samples), eigenvals_selected is (n_components,)
        # We need to multiply each row of X_transformed by the corresponding sqrt(eigenvalue)
        scaled_X = X_transformed * np.sqrt(eigenvals_selected)[:, None]  # Broadcasting fix
        K_reconstructed = eigenvecs_selected @ scaled_X
        
        # This gives us the kernel representation, but we need to map back to input space
        # This is the pre-image problem - we'll use a simple approximation
        
        # For RBF kernel, we can approximate by finding the closest training samples
        if self.kernel == 'rbf':
            # Find closest matches in kernel space
            distances = np.sum((self.K_ - K_reconstructed[:, :, None]) ** 2, axis=0)
            closest_indices = np.argmin(distances, axis=1)
            X_reconstructed = self.X_[:, closest_indices]
        else:
            # For linear kernel, we can do better
            # K = X^T @ X, so we can use pseudoinverse
            X_reconstructed = np.linalg.pinv(self.X_) @ K_reconstructed
        
        return X_reconstructed
    
    def get_compression_ratio(self) -> float:
        """
        Get compression ratio achieved by dimensionality reduction
        
        Returns:
        --------
        ratio : float
            Compression ratio (original_size / compressed_size)
        """
        if self.n_components is None:
            return 1.0
        
        original_size = self.X_.shape[0] * self.X_.shape[1]
        compressed_size = self.n_components * self.X_.shape[1]
        
        return original_size / compressed_size
    
    def explained_variance_ratio(self, n_components: Optional[int] = None) -> np.ndarray:
        """
        Get explained variance ratio for components
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to consider
            
        Returns:
        --------
        ratios : np.ndarray
            Explained variance ratio for each component
        """
        if self.eigenvalues_ is None:
            raise ValueError("Must fit model before computing explained variance")
        
        if n_components is None:
            n_components = len(self.eigenvalues_)
        
        total_var = np.sum(self.eigenvalues_)
        return self.eigenvalues_[:n_components] / total_var
    
    def plot_convergence(self):
        """Plot convergence curve"""
        if not self.objective_values_:
            print("No convergence data available")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.objective_values_)
        plt.title('RKPCA Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
    
    def score(self, M: np.ndarray) -> float:
        """
        Compute relative recovery error
        
        Parameters:
        -----------
        M : np.ndarray
            Original data matrix
            
        Returns:
        --------
        error : float
            Relative Frobenius norm error ||X - X_true||_F / ||X_true||_F
        """
        if self.X_ is None:
            raise ValueError("Must fit model before computing score")
        
        return np.linalg.norm(M - self.X_, 'fro') / np.linalg.norm(M, 'fro')


def generate_synthetic_data(d=20, r=2, n=100, noise_density=0.3, noise_std=1.0, 
                          random_state=None):
    """
    Generate synthetic nonlinear data for testing RKPCA
    
    Parameters:
    -----------
    d : int, default=20
        Data dimension
    r : int, default=2  
        Latent dimension
    n : int, default=100
        Number of samples
    noise_density : float, default=0.3
        Fraction of entries corrupted by sparse noise
    noise_std : float, default=1.0
        Standard deviation of sparse noise
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    M : np.ndarray, shape (d, n)
        Corrupted data matrix
    X_true : np.ndarray, shape (d, n)
        Clean data matrix  
    E_true : np.ndarray, shape (d, n)
        Sparse noise matrix
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate latent variables
    Z = np.random.uniform(-1, 1, (r, n))
    
    # Generate projection matrices
    P1 = np.random.randn(d, r)
    P2 = np.random.randn(d, r) 
    P3 = np.random.randn(d, r)
    
    # Generate nonlinear data: X = P1*Z + 0.5*(P2*Z^2 + P3*Z^3)
    X_true = P1 @ Z + 0.5 * (P2 @ (Z**2) + P3 @ (Z**3))
    
    # Generate sparse noise
    E_true = np.zeros_like(X_true)
    mask = np.random.rand(d, n) < noise_density
    E_true[mask] = np.random.normal(0, noise_std, np.sum(mask))
    
    # Corrupted data
    M = X_true + E_true
    
    return M, X_true, E_true


# Example usage and demonstration
if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic nonlinear data...")
    M, X_true, E_true = generate_synthetic_data(d=7690, r=2, n=10, 
                                              noise_density=0.3, 
                                              random_state=42)
    
    print(f"Data shape: {M.shape}")
    print(f"True rank of clean data: {np.linalg.matrix_rank(X_true)}")
    print(f"Noise density: {np.mean(E_true != 0):.2%}")
    
    # Example 1: Standard RKPCA decomposition
    print("\n" + "="*50)
    print("EXAMPLE 1: Standard RKPCA Decomposition")
    print("="*50)
    
    rkpca = RKPCA(kernel='rbf', lambda_0=0.5, max_iter=50, 
                  algorithm='plm_adss', verbose=True)
    
    X_recovered, E_recovered = rkpca.fit_transform(M)
    
    # Evaluate results
    recovery_error = np.linalg.norm(X_true - X_recovered, 'fro') / np.linalg.norm(X_true, 'fro')
    noise_error = np.linalg.norm(E_true - E_recovered, 'fro') / np.linalg.norm(E_true, 'fro')
    
    print(f"\nRecovery Results:")
    print(f"Clean data recovery error: {recovery_error:.4f}")
    print(f"Noise recovery error: {noise_error:.4f}")
    print(f"Reconstruction error: {np.linalg.norm(M - X_recovered - E_recovered, 'fro'):.6f}")
    
    # Example 2: RKPCA with compression
    print("\n" + "="*50)
    print("EXAMPLE 2: RKPCA with Data Compression")
    print("="*50)
    
    # Try different compression levels
    n_components_list = [8, 16, 32, 64, 128, 256]
    
    for n_comp in n_components_list:
        print(f"\n--- Compressing to {n_comp} components ---")
        
        # Fit with compression
        rkpca_compressed = RKPCA(n_components=n_comp, kernel='rbf', lambda_0=0.5, 
                               max_iter=50, verbose=False)
        
        # Get compressed representation
        X_compressed = rkpca_compressed.fit_transform(M, n_components=n_comp)
        
        print(f"Original shape: {M.shape}")
        print(f"Compressed shape: {X_compressed.shape}")
        print(f"Compression ratio: {rkpca_compressed.get_compression_ratio():.2f}x")
        
        # Explained variance
        explained_var = np.sum(rkpca_compressed.explained_variance_ratio(n_comp))
        print(f"Explained variance: {explained_var:.4f}")
        
        # Try to reconstruct (approximate)
        try:
            X_reconstructed = rkpca_compressed.inverse_transform(X_compressed)
            reconstruction_error = np.linalg.norm(rkpca_compressed.X_ - X_reconstructed, 'fro') / np.linalg.norm(rkpca_compressed.X_, 'fro')
            print(f"Reconstruction error: {reconstruction_error:.4f}")
        except Exception as e:
            print(f"Reconstruction failed: {e}")
    
    # Example 3: Compare with sklearn KernelPCA
    print("\n" + "="*50)
    print("EXAMPLE 3: Comparison with sklearn KernelPCA")
    print("="*50)
    
    try:
        from sklearn.decomposition import KernelPCA
        
        # Fit sklearn KernelPCA on clean data (for comparison)
        n_comp = 5
        kpca_sklearn = KernelPCA(n_components=n_comp, kernel='rbf', gamma=1/(2*rkpca.sigma))
        X_sklearn = kpca_sklearn.fit_transform(X_true.T).T  # sklearn expects (n_samples, n_features)
        
        # Our robust version on noisy data
        rkpca_robust = RKPCA(n_components=n_comp, kernel='rbf', sigma=rkpca.sigma, 
                           lambda_0=0.5, verbose=False)
        X_robust = rkpca_robust.fit_transform(M, n_components=n_comp)
        
        print(f"Sklearn KernelPCA (clean data) shape: {X_sklearn.shape}")
        print(f"Robust KPCA (noisy data) shape: {X_robust.shape}")
        
        print(f"Sklearn explained variance: {np.sum(kpca_sklearn.eigenvalues_[:n_comp]) / np.sum(kpca_sklearn.eigenvalues_):.4f}")
        print(f"Robust KPCA explained variance: {np.sum(rkpca_robust.explained_variance_ratio(n_comp)):.4f}")
        
    except ImportError:
        print("sklearn not available for comparison")
    
    # Example 4: Transform new data
    print("\n" + "="*50)
    print("EXAMPLE 4: Transform New Data")
    print("="*50)
    
    # Generate new test data
    M_test, X_test_true, E_test_true = generate_synthetic_data(d=7690, r=2, n=10, 
                                                             noise_density=0.2, 
                                                             random_state=123)
    
    # Transform using fitted model
    X_test_compressed = rkpca_compressed.transform(M_test, n_components=5)
    print(f"Test data original shape: {M_test.shape}")
    print(f"Test data compressed shape: {X_test_compressed.shape}")
    
    # Plot convergence for the first example
    print("\n" + "="*50)
    print("Convergence Plot")
    print("="*50)
    rkpca.plot_convergence()