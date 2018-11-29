import torch

def torch_randomized_svd(X, max_r=None, k=50):
    A = X.clone()
    if A.is_cuda:
        device = A.get_device()
    m, n = A.shape
    if max_r is None:
        max_r = min(m, n)
    res_U = torch.zeros(m, max_r, dtype=A.dtype)
    res_Sigma = torch.zeros(max_r, dtype=A.dtype)
    res_V = torch.zeros(n, max_r, dtype=A.dtype)
    idx = 0
    with torch.no_grad():
        if max_r is None:
            max_r = min(m, n)
        while max_r:
            k = min(k, max_r)
            Omega = torch.randn(n, k, dtype = A.dtype, device=A.device)
            Y = A @ Omega
            Q, R = torch.qr(Y)
            B = Q.cpu().t() @ A.cpu()
            Uhat, Sigma, V = torch.svd(B)
            U = Q @ (Uhat.to(Q.device))
            approx = U @ (torch.diag(Sigma).to(U.device))
            approx = approx @ (V.t().to(approx.device))
            A -= approx
            max_r -= k
            res_U[:, idx:idx+k] = U
            res_Sigma[idx:idx+k] = Sigma
            res_V[:, idx:idx+k] = V
            idx += k
    return res_U, res_Sigma, res_V