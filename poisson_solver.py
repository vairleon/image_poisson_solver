import torch
import torch.nn as nn
import torch.nn.functional as F
from cg import cg_batch

class PoissonSolver:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', debug=False):
        """
        init the poisson solver
        
        Args:
            device: the device to run the solver, default is GPU (if available)
        """
        self.device = device
        self.debug = debug
        print(f"Using device: {self.device}")
    
    def sparse_laplacian_matrix(self, h, w):
        """
        build the sparse laplacian matrix
        
        Args:
            h: the height of the image
            w: the width of the image
            
        Returns:
            the sparse laplacian matrix (for conjugate gradient method)
        """
        n = h * w
        
        # create the diagonal elements
        diag = torch.ones(n, device=self.device) * -4
        
        # create the horizontal adjacent elements
        horiz = torch.ones(n-1, device=self.device)
        
        # handle the boundary conditions: the points in different rows should not be connected
        for i in range(w-1, n-1, w):
            horiz[i] = 0
        
        # create the vertical adjacent elements
        vert = torch.ones(n-w, device=self.device)
        
        # build the indices and values of the sparse matrix
        i = torch.arange(n, device=self.device)
        j = torch.arange(n, device=self.device)
        v = diag
        
        # add the horizontal adjacent elements
        i_horiz = torch.cat([torch.arange(n-1, device=self.device), 
                             torch.arange(1, n, device=self.device)])
        j_horiz = torch.cat([torch.arange(1, n, device=self.device), 
                             torch.arange(n-1, device=self.device)])
        v_horiz = torch.cat([horiz, horiz])
        
        # add the vertical adjacent elements
        i_vert = torch.cat([torch.arange(n-w, device=self.device), 
                            torch.arange(w, n, device=self.device)])
        j_vert = torch.cat([torch.arange(w, n, device=self.device), 
                            torch.arange(n-w, device=self.device)])
        v_vert = torch.cat([vert, vert])
        
        # merge all the indices and values
        i = torch.cat([i, i_horiz, i_vert])
        j = torch.cat([j, j_horiz, j_vert])
        v = torch.cat([v, v_horiz, v_vert])
        
        # create the sparse matrix
        indices = torch.stack([i, j])
        L = torch.sparse_coo_tensor(indices, v, (n, n), device=self.device)
        
        return L    
    
    def dense_laplacian_matrix(self, h, w):
        """corrected dense laplacian matrix construction"""
        n = h * w
        L = torch.zeros((n, n), device=self.device)
        
        # directly use the sparse matrix operator
        indices = torch.zeros((2, 5*n), device=self.device, dtype=torch.long)
        values = torch.zeros(5*n, device=self.device)
        
        # the diagonal elements
        idx = 0
        for i in range(n):
            indices[0, idx] = i
            indices[1, idx] = i
            values[idx] = -4.0
            idx += 1
            
            # the upper element
            if i >= w:
                indices[0, idx] = i
                indices[1, idx] = i-w
                values[idx] = 1.0
                idx += 1
            
            # the lower element
            if i + w < n:
                indices[0, idx] = i
                indices[1, idx] = i+w
                values[idx] = 1.0
                idx += 1
            
            # the left element (not across rows)
            if i % w != 0:
                indices[0, idx] = i
                indices[1, idx] = i-1
                values[idx] = 1.0
                idx += 1
            
            # the right element (not across rows)
            if (i+1) % w != 0:
                indices[0, idx] = i
                indices[1, idx] = i+1
                values[idx] = 1.0
                idx += 1
        
        # only keep the valid indices
        indices = indices[:, :idx]
        values = values[:idx]
        
        sparse_L = torch.sparse_coo_tensor(indices, values, (n, n), device=self.device)
        return sparse_L.to_dense()
    
    
    def apply_laplacian(self, image:torch.Tensor):
        """
        convolute the image with the Laplacian kernel
        Args:
            image: tensor of shape (h, w)
        Returns:
            laplacian: Laplacian result
        """
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        
        # 定义Laplacian卷积核
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=torch.float32)
        
        # 
        kernel = laplacian_kernel.view(1, 1, 3, 3).to(image.device)
        
        # 
        _, _, H, W = image.shape
        laplacian = []
        
        # per channel
        padded = F.pad(image, (1,1,1,1), mode='reflect')
        laplacian = F.conv2d(padded, kernel, padding=0)
        
        #
        return laplacian.squeeze()

    
    def conjugate_gradient(self, A_fn, b, x0=None, tol=1e-7, max_iter=5000):
        """
        Enhanced conjugate gradient method with improved stability and accuracy
        
        Args:
            A_fn: function, apply matrix A to vector
            b: right vector
            x0: initial guess
            tol: convergence tolerance
            max_iter: maximum number of iterations
            
        Returns:
            approximate solution, convergence info
        """
        n = b.numel()
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()
            
        # 计算初始残差 r = b - Ax
        r = b - A_fn(x)
        p = r.clone()
        
        # 计算初始范数
        b_norm = torch.norm(b)
        r_norm = torch.norm(r)
        initial_residual = r_norm
        
        # 避免除零
        eps = torch.finfo(b.dtype).eps
        if b_norm < eps:
            b_norm = 1.0
            
        # 相对残差作为收敛判据
        rel_tol = tol * b_norm
        rdotr = torch.dot(r, r)
        
        # 记录收敛历史
        residual_history = [r_norm.item()]
        
        for i in range(max_iter):
            Ap = A_fn(p)
            
            # 计算步长
            pAp = torch.dot(p, Ap)
            if abs(pAp) < eps:
                if self.debug:
                    print(f"Warning: pAp near zero at iteration {i+1}")
                break
                
            alpha = rdotr / pAp
            
            # 更新解和残差
            x_new = x + alpha * p
            
            # 每k次迭代重新计算残差，避免误差累积
            if (i + 1) % 50 == 0:
                r = b - A_fn(x_new)
            else:
                r = r - alpha * Ap
                
            rdotr_new = torch.dot(r, r)
            beta = rdotr_new / rdotr
            
            # 更新搜索方向
            p = r + beta * p
            
            # 每k次迭代进行一次Gram-Schmidt正交化
            if (i + 1) % 50 == 0:
                p = r + beta * p - torch.dot(p, r) * r / rdotr_new
            
            # 更新变量
            x = x_new
            rdotr = rdotr_new
            r_norm = torch.sqrt(rdotr)
            residual_history.append(r_norm.item())
            
            # 检查收敛性
            if r_norm < rel_tol:
                if self.debug:
                    print(f"CG converged after {i+1} iterations")
                    print(f"Final relative residual: {r_norm/b_norm:.2e}")
                break
                
            # 监控收敛性
            if (i+1) % 100 == 0 and self.debug:
                rel_res = r_norm/b_norm
                print(f"Iteration {i+1}, relative residual: {rel_res:.2e}")
                
                # 检查是否停滞
                if i > 100:
                    recent_progress = abs(residual_history[-1] - residual_history[-100]) / residual_history[-100]
                    if recent_progress < tol * 0.01:
                        print(f"Warning: Convergence stagnated at iteration {i+1}")
                        break
            
            # 检查数值稳定性
            if torch.isnan(r_norm) or torch.isinf(r_norm):
                print("Warning: Numerical instability detected")
                break
        
        if i == max_iter - 1:
            print(f"Warning: CG did not converge after {max_iter} iterations")
            print(f"Final relative residual: {r_norm/b_norm:.2e}")
            
        # 最终残差检查
        true_residual = torch.norm(b - A_fn(x))
        if self.debug:
            print(f"True final relative residual: {true_residual/b_norm:.2e}")
            
        convergence_info = {
            'iterations': i + 1,
            'initial_residual': initial_residual.item(),
            'final_residual': r_norm.item(),
            'true_residual': true_residual.item(),
            'residual_history': residual_history,
            'converged': i < max_iter - 1
        }
            
        return x, convergence_info
    
    
    def conjugate_gradient_debug(self, A_fn, b, x0=None, tol=1e-7, max_iter=5000):
        """
        implement the conjugate gradient method to solve the linear system
        
        Args:
            A: function
            b: right vector
            x0: initial guess
            tol: convergence tolerance
            max_iter: maximum number of iterations
            
        Returns:
            approximate solution
        """
        
        n = b.numel()
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()
        
        r = b.clone()
        p = b.clone()
        
        # r = b - A_fn(x)
        # p = r.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(max_iter):
            
            Ap = A_fn(p)
            alpha = rdotr / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            
            rdotr_new = torch.dot(r, r)
            
            if torch.sqrt(rdotr_new) < tol:
                print(f"CG converged after {i+1} iterations")
                break
                
            p = r + (rdotr_new / rdotr) * p
            rdotr = rdotr_new
            
            if (i+1) % 100 == 0:
                print(f"iteration {i+1}, residual: {torch.sqrt(rdotr_new).item()}")
        
        if i == max_iter - 1:
            print(f"Warning: CG did not converge after {max_iter} iterations, residual: {torch.sqrt(rdotr_new).item()}")
            
        return x
    
    def lu_solve(self, A:torch.Tensor, b:torch.Tensor):
        """
        solve the linear system Ax = b using matrix decomposition
        
        Args:
            A: coefficient matrix
            b: right vector
            
        Returns:
            exact solution
        """
        # use the solve function of PyTorch (based on LU decomposition)
        x = torch.linalg.solve(A, b.unsqueeze(1))
        return x.squeeze(1)
    
    def cholesky_solve(self, A:torch.Tensor, b:torch.Tensor):
        """
        solve the linear system Ax = b using modified Cholesky decomposition
        """
        n = A.shape[0]
        
        try:
            # solve cholesky decomposition
            L = torch.linalg.cholesky(A)
            x = torch.cholesky_solve(b.view(-1, 1), L)
            
        except Exception as e:
            # !DEBUG: still incorrect
            min_eig = torch.min(A.diagonal())
            
            if min_eig <= 0:
                # 
                shift = abs(min_eig) + 1e-6
                D = torch.diag(A.diagonal())
                D_mod = torch.where(D.diagonal() <= shift, 
                                  shift * torch.ones_like(D.diagonal()), 
                                  D.diagonal())
                
                # 
                A_mod = A.clone()
                A_mod.diagonal().copy_(D_mod)
                
                # 
                L = torch.linalg.cholesky(A_mod)
                x = torch.cholesky_solve(b.view(-1, 1), L)
                
                # 
                r = b - (A @ x).squeeze()
                dx = torch.cholesky_solve(r.view(-1, 1), L)
                x = x + dx
            
        return x.squeeze()
    
    
    def solve(self, f:torch.Tensor, boundary:torch.Tensor, boundary_values:torch.Tensor, interior_mask:torch.Tensor=None, method='cg'):
        """
        solve the Poisson equation: ∇²u = f, with Dirichlet boundary conditions
        
        Args:
            f: right function, shape [h, w] torch.Tensor
            boundary: boundary condition mask, shape [h, w], 1 represents boundary points
            boundary_values: boundary values, shape [h, w]
            method: 'cg' represents conjugate gradient method
            
        Returns:
            solution u, shape [h, w]
        """
        assert torch.is_tensor(f)
        assert torch.is_tensor(boundary)
        assert torch.is_tensor(boundary_values)
        #
        f = f.to(self.device)
        boundary = boundary.to(self.device)
        boundary_values = boundary_values.to(self.device)
        #
        h, w = f.shape
        
        # initialize the solution
        u = boundary_values.clone()

        # the index of the interior points
        if interior_mask is None:
            interior_mask = 1 - boundary
        
        #            
        b = f * interior_mask + boundary_values * boundary
        b_flat = b.reshape(-1)
        
        #
        if self.debug:
            pass
        
        # fast and memory efficient (avoid building full A matrix)
        if method == 'cg':
            
             # build the Laplacian matrix
            # print("build the Laplacian matrix...")
            # A = self.dense_laplacian_matrix(h, w)
            # #
            # n = h * w
            # indices = torch.arange(n, device=self.device)
            
            # # modify the matrix to apply the boundary conditions
            # boundary_flat = boundary.reshape(-1)
            # boundary_indices = indices[boundary_flat > 0]
            # A[boundary_indices, :] = 0
            # A[boundary_indices, boundary_indices] = 1
            # #
            
            # set function to apply the Laplacian
            def A_fn(x_flat):
                x = x_flat.reshape(f.shape)
                # 
                x_full = x  #* interior_mask
                # apply Laplacian
                Ax = self.apply_laplacian(x_full)
                # compute Ax
                return (Ax * interior_mask + x_full * boundary).reshape(1, -1, 1)
                # return torch.bmm(A.unsqueeze(0), x_flat)
            
            # solve the system
            # u_interior_flat = self.conjugate_gradient_debug(A_fn, b_flat)
            # u_interior_flat, convergence_info = self.conjugate_gradient(A_fn, b_flat)
            u_interior_flat, convergence_info = cg_batch(A_fn, b_flat.view(1, -1, 1), verbose=self.debug)
 
            u_interior = u_interior_flat.reshape(f.shape)
            
            # merge the interior solution and boundary values
            u =  u + u_interior * interior_mask
            
        # direct solve or Cholesky decomposition
        elif method == 'lu' or method == 'cholesky':
            n = h * w
            
            # build the Laplacian matrix
            print("build the Laplacian matrix...")
            A = self.dense_laplacian_matrix(h, w)
            #
            indices = torch.arange(n, device=self.device)
            
            # modify the matrix to apply the boundary conditions
            boundary_flat = boundary.reshape(-1)
            boundary_indices = indices[boundary_flat > 0]
            A[boundary_indices, :] = 0
            A[boundary_indices, boundary_indices] = 1
            
            if self.debug:
                print("Matrix condition number:", torch.linalg.cond(A).item())
                print("Matrix rank:", torch.linalg.matrix_rank(A).item())
                
            #
            print(f"solve the linear system ({n}x{n})...")
            
            # choose the solver
            if method == 'lu':
                print("use lu decomposition...")
                u_flat = self.lu_solve(A, b_flat)
            else:  # cholesky
                #
                print("use Cholesky decomposition...")
                u_flat = self.cholesky_solve(A, b_flat)
            
            # merge the interior solution and boundary values
            u_interior = u_flat.reshape(h, w)
            u = u + u_interior * interior_mask

        return u
    
    def visualize(self, u, title="Poisson方程解", cmap='viridis'):
        """
        visualize the solution
        
        Args:
            u: the solution vector
            title: the title of the image
            cmap: the color map
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        if torch.is_tensor(u):
            u = u.cpu().numpy()
            
        plt.figure(figsize=(10, 8))
        plt.imshow(u, cmap=cmap)
        plt.colorbar(label='value')
        plt.title(title)
        plt.tight_layout()
        plt.show()


# Unit test
if __name__ == "__main__":
    # create a test problem
    h, w = 16, 16
    
    # create a simple test image as the right function
    f = torch.zeros((h, w), dtype=torch.float32)
    f[h//4:3*h//4, w//4:3*w//4] = 0.25
    
    # define the boundary mask and boundary values
    boundary = torch.zeros((h, w), dtype=torch.float32)
    boundary[0, :] = 1
    boundary[-1, :] = 1
    boundary[:, 0] = 1
    boundary[:, -1] = 1
    
    boundary_values = torch.zeros((h, w), dtype=torch.float32)
    boundary_values[0, :] = 0.5  # the upper boundary is 0.5
    
    
    # create the solver and solve the problem
    solver = PoissonSolver(debug=True)
    f = solver.apply_laplacian(f)
    
    #
    solution = solver.solve(f, boundary, boundary_values, method='cg')
    solver.visualize(solution, "Conjugate Gradient Solution")
    #
    solution = solver.solve(f, boundary, boundary_values, method='lu')
    solver.visualize(solution, "Lu Decomposition Solution")
    
    solution = solver.solve(f, boundary, boundary_values, method='cholesky')
    solver.visualize(solution, "Cholesky Decomposition Solution")
    
    # # test the performance of the solver
    import time
    solver = PoissonSolver(debug=False)
    
    start_time = time.time()
    solution = solver.solve(f, boundary, boundary_values, method='cg')
    end_time = time.time()
    print(f"cg cost: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    solution = solver.solve(f, boundary, boundary_values, method='lu')
    end_time = time.time()
    print(f"lu cost: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    solution = solver.solve(f, boundary, boundary_values, method='cholesky')
    end_time = time.time()
    print(f"cholesky cost: {end_time - start_time:.4f} seconds")
    
