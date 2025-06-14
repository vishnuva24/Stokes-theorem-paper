import numpy as np
import matplotlib.pyplot as plt


def newton_preimages(w):
    """
    For Newton map of P(z)=z^3-1,
      N(z) = z - (z^3-1)/(3z^2) = (2z^3+1)/(3z^2)
    solve for N(z)=w  => 2 z^3 - 3w z^2 + 1 = 0
    return all three roots z.
    """
    coeffs = [2, -3*w, 0, 1]
    return np.roots(coeffs)

def plot_newton_preimage_boundary(root,
                                  radius=0.02,
                                  num_seeds=720,
                                  levels=6):
    """
    starts with a small circle around an sttracting fixed point (one cube-root of unity),
    then for `levels` times replace every boundary point by *all* its
    polynomial-newton-map preimages. 
    """

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    angles = np.linspace(0, 2*np.pi, num_seeds, endpoint=False)
    boundary = [root + radius * np.exp(1j*θ) for θ in angles]


    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # the attracting fixed point
    ax.plot(root.real, root.imag, 'o', color='black')
    for i in range(levels):
        xs = [z.real for z in boundary]
        ys = [z.imag for z in boundary]
        color = colors[i % len(colors)]
        ax.scatter(xs, ys, s=0.3, color=color, label=f'Level {i+1}')
        ax.set_title(f'Newton Preimage Boundary – Level {i+1}/{levels}')
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.axis('equal')
        ax.legend(markerscale=5, fontsize='small', loc='upper right')
        # draw and pause to show in real time
        fig.canvas.draw()
        plt.pause(0.5)
        # compute next preimage boundary
        next_bdry = []
        for w in boundary:
            next_bdry.extend(newton_preimages(w))
        boundary = next_bdry

    plt.show()
    
if __name__ == "__main__":
    # compute the three cube‐roots of unity (fixed points of Newton map)
    roots = [1,
             np.exp(2j*np.pi/3),
             np.exp(4j*np.pi/3)]

    # pick one root and plot its backward‐preimage boundary
    plot_newton_preimage_boundary(
        root=roots[0],
        radius=0.02,
        num_seeds=360,
        levels=6
    )