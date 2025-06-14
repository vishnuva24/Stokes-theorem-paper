import numpy as np
import matplotlib.pyplot as plt

def generate_julia(coeffs,
                   xlim=(-1.5, 1.5),
                   ylim=(-1.5, 1.5),
                   res=1000,
                   max_iter=200,
                   escape_radius=2.0):
    """
    coeffs: list of polynomial coefficients [a_n, a_{n-1}, ..., a_0]
            representing f(z) = a_n*z^n + ... + a_0.
    xlim, ylim: real/imag range tuples
    res: resolution (pixels) per axis
    max_iter: maximum number of iterations
    escape_radius: modulus threshold for escape
    """
    # build the grid of initial z0 values
    xs = np.linspace(xlim[0], xlim[1], res)
    ys = np.linspace(ylim[0], ylim[1], res)
    Z0 = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]
    Z = Z0.copy()
    # iteration counts (0 means “never escaped”)
    counts = np.zeros(Z.shape, dtype=int)
    # Horner’s method polynomial coefficients
    A = np.array(coeffs, dtype=complex)

    for i in range(1, max_iter + 1):
        # compute f(Z) via Horner’s method
        W = A[0] + 0*Z
        for a in A[1:]:
            W = W * Z + a
        Z = W
        # mark points that escape on this iteration
        escaped = (np.abs(Z) > escape_radius) & (counts == 0)
        counts[escaped] = i

    return counts

def plot_julia(counts, xlim, ylim, cmap='cool'):
    plt.figure(figsize=(8,8))
    plt.imshow(counts,
               extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
               origin='lower',
               cmap=cmap,
               interpolation='bilinear')
    plt.colorbar(label='Escape iteration')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Julia set')
    plt.show()

def plot_filled_julia(counts, xlim, ylim, cmap='binary'):
    """
    Display the filled Julia set: points with counts==0 never escaped.
    """
    mask = (counts == 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(mask,
               extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
               origin='lower',
               cmap=cmap)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Filled Julia Set')
    plt.show()

if __name__ == "__main__":
    # Example for f(z)=z^2 + (0.285+0.01j):
    coeffs = [1, 0.7j,0]
    xlim, ylim = (-1.5, 1.5), (-1.5, 1.5)
    counts = generate_julia(coeffs,
                            xlim=xlim,
                            ylim=ylim,
                            res=1200,
                            max_iter=300,
                            escape_radius=2.0)
    # plot the filled Julia set (points that never escaped)
    plot_filled_julia(counts, xlim, ylim)