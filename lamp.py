import numpy as np

def LAMP(X, Xs, Ys, tol=1e-9):
    """Local Affine Multidimensional Projection (LAMP) is a multidimensional
    projection technique that uses control points located in the projected
    space to guide the positioning of the remaining data.

    Args:
        X (numpy.array): The input matrix of size NxM. N is the number of
        objects and M is the original dimension

        Xs (numpy.array): The control points in their original space.

        Ys (numpy.array): A CxP matrix with the control points positioned in
        the projected space. The value of C must match `len(Xs)` and P is the
        number of dimensions of the projected space.

    KwArgs:
        tol (numeric): The numeric tolerance. Default value is 1e-9

    Returns:
        numpy.array: A NxP matrix, where each row is a single projected point.
        The control points are not included in the results.
    """

    try:
        dim1, dim2 = X.shape
    except ValueError:
        dim1 = 1
        dim2 = len(X)

    Y = np.zeros(shape=(dim1, Ys.shape[1]),
                 dtype=np.double)
    alphas = np.zeros(shape=Xs.shape[0], dtype=np.double)

    for i in range(dim1):
        # Calculating the alphas (Eq. 2).
        for j in range(Xs.shape[0]):
            alphas[j] = 1 / max(np.linalg.norm(Xs[j] - X[i]), tol)

        sum_alpha = alphas.sum()

        # Calculating x.tilde and y.tilde (Eq. 3).
        x_tilde = np.zeros(dim2)
        y_tilde = np.zeros(Y.shape[1])

        for j in range(Ys.shape[0]):
            x_tilde += alphas[j] * Xs[j]
            y_tilde += alphas[j] * Ys[j]

        x_tilde /= sum_alpha
        y_tilde /= sum_alpha

        # Calculating x.hat and y.hat (Eq. 4).
        x_hat = np.zeros(shape=(Xs.shape[0], dim2))
        y_hat = np.zeros(shape=Ys.shape)
        for j in range(Ys.shape[0]):
            x_hat[j] = Xs[j] - x_tilde
            y_hat[j] = Ys[j] - y_tilde

        # Calculating A and B (Eq. 6).
        A = np.zeros(shape=(Xs.shape[0], dim2), dtype=np.double)
        B = np.zeros(shape=(Xs.shape[0], Ys.shape[1]), dtype=np.double)
        for j in range(Ys.shape[0]):
            A[j] = np.sqrt(alphas[j]) * x_hat[j]
            B[j] = np.sqrt(alphas[j]) * y_hat[j]

        # SVD decomposition of A.T * B (Eq. 7).
        U, D, V = np.linalg.svd(np.dot(A.T, B))

        # Orthogonal transform matrix (Eq. 8).
        M = np.dot(U[:, 0:Ys.shape[1]], V)

        # Actual projection of the data.
        Y[i] = np.dot(X[i] - x_tilde, M) + y_tilde

    return Y
