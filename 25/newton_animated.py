import random
import cmath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def newton_preimages(w):
    """
    For Newton map of P(z)=z^3−1,
      N(z) = z − (z^3−1)/(3z^2) = (2z^3+1)/(3z^2)
    solve N(z)=w  ⇒ 2 z^3 − 3w z^2 + 1 = 0
    return all three roots z.
    """
    coeffs = [2, -3*w, 0, 1]
    return np.roots(coeffs)

def plot_newton_preimage_boundary(root,
                                  radius=0.02,
                                  num_seeds=720,
                                  levels=6,
                                  steps_per_level=10,
                                  interval=200):
    """
    Smooth animation of Newton‐map preimage boundary:
    start on small circle around `root` and over `levels`
    iteratively take all preimages. Points split smoothly
    from parent to their three children, *keeping* all
    previous boundaries visible.
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    angles = np.linspace(0, 2*np.pi, num_seeds, endpoint=False)
    init = root + radius * np.exp(1j*angles)

    # precompute all boundary levels
    boundary_levels = [init]
    for _ in range(levels):
        prev = boundary_levels[-1]
        nxt = []
        for w in prev:
            nxt.extend(newton_preimages(w))
        boundary_levels.append(np.array(nxt))

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
    ax.set_title('Newton Preimage Boundary Animation')

    # one scatter artist per level
    scatters = []
    for lvl in range(levels+1):
        pts = boundary_levels[lvl] if lvl == 0 else np.empty((0,2))
        xy = np.column_stack((pts.real, pts.imag))
        scat = ax.scatter(xy[:,0], xy[:,1],
                          s=0.5,
                          color=colors[lvl % len(colors)],
                          label=f'Level {lvl}')
        scatters.append(scat)
    ax.legend(markerscale=5, fontsize='small', loc='upper right')

    total_frames = levels * steps_per_level + 1

    def init_anim():
        # explicitly draw the 0th‐level boundary
        for i, scat in enumerate(scatters):
            if i == 0:
                pts = boundary_levels[0]
                scat.set_color(colors[0])
            else:
                pts = np.empty((0,2))
            xy = (np.column_stack((pts.real, pts.imag))
                  if pts.size else np.empty((0,2)))
            scat.set_offsets(xy)
        return scatters

    def animate(frame):
        if frame == 0:
            return scatters
        idx = frame - 1
        lvl = idx // steps_per_level
        t = ((idx % steps_per_level) + 1) / steps_per_level
        P = boundary_levels[lvl]
        C = boundary_levels[lvl+1]
        # interpolate each parent to its 3 children
        P_rep = np.repeat(P, 3)
        interp = (1 - t)*P_rep + t*C

        # update each scatter, coloring the moving boundary in the *target* color
        for i, scat in enumerate(scatters):
            if i < lvl:
                pts = boundary_levels[i+1] if i < levels else np.empty((0,2))
                scat.set_color(colors[(i) % len(colors)])
            elif i == lvl:
                pts = interp
                # use the color of the next level
                scat.set_color(colors[(lvl) % len(colors)])
            else:
                pts = np.empty((0,2))
            xy = (np.column_stack((pts.real, pts.imag))
                  if pts.size else np.empty((0,2)))
            scat.set_offsets(xy)
        return scatters

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_anim,
        frames=total_frames,
        interval=interval,
        blit=True
    )
    # save animation as GIF (requires pillow)
    anim.save('newton_preimage_boundary.gif', writer='pillow', fps=10)

    plt.show()

if __name__ == "__main__":
    # compute the three cube‐roots of unity (fixed points of Newton map)
    roots = [1,
             np.exp(2j*np.pi/3),
             np.exp(4j*np.pi/3)]

    # pick one root and animate its backward‐preimage boundary
    plot_newton_preimage_boundary(
        root=roots[0],
        radius=0.02,
        num_seeds=360,
        levels=6,
        steps_per_level=15,
        interval=100
    )