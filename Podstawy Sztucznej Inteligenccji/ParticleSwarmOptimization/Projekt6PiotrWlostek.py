import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return ((1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2)

num_of_particles = 50
num_of_iterations = 40
X = np.random.rand(2,num_of_particles)*9-4.5 # inicjalizacja cząstek na przesytrzeni [-4.5, 4.5]x[-4.5, 4.5]
V = np.random.randn(2, num_of_particles) * 0.1 #początkowe wektory prędkości 
p_bests = X.copy()
p_bests_scores = f(X[0], X[1])
global_best = p_bests[:, p_bests_scores.argmin()]
global_best_score = p_bests_scores.min()
c1 = 0.4
c2 = 0.6
w = 0.7


def PSO():
    global X, V, p_bests, p_bests_scores, global_best, global_best_score
    r = np.random.rand(2)
    V = w * V + c1*r[0]*(p_bests - X) + c2*r[1]*(global_best.reshape(-1,1) - X)
    X += V

    particles_out_of_bound = (X < -4.5) | (X > 4.5)
    X[particles_out_of_bound] = np.clip(X[particles_out_of_bound], -4.5, 4.5)
    V[particles_out_of_bound] *= -0.5

    new_scores = f(X[0], X[1])
    particles_with_better_positions = new_scores < p_bests_scores
    p_bests[:,particles_with_better_positions] = X[:, particles_with_better_positions]
    p_bests_scores[particles_with_better_positions] = new_scores[particles_with_better_positions]

    if new_scores.min() < global_best_score:
        global_best = X[:, new_scores.argmin()]
        global_best_score = new_scores.min()


def animate_heatmap():
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(-4.5, 4.5, 100)
    y = np.linspace(-4.5, 4.5, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = f(X_grid, Y_grid)

    def init():
        image = ax.imshow(Z, extent=[-4.5, 4.5, -4.5, 4.5], origin='lower', cmap='viridis', alpha=0.7)
        ax.set_title('PSO Convergence')
        return image,

    def update(frame):
        global X, V, p_bests, p_bests_scores, global_best, global_best_score
        PSO()
        ax.clear()
        ax.imshow(Z, extent=[-4.5, 4.5, -4.5, 4.5], origin='lower', cmap='viridis', alpha=0.7)
        ax.scatter(X[0], X[1], color='red', s=50, alpha=0.6)
        ax.set_title(f'Iteration {frame + 1}: Global Best = {global_best_score:.4f}')
        return ax,

    ani = FuncAnimation(fig, update, frames=num_of_iterations, interval = 160, init_func=init, blit=False, repeat=False)
    plt.show()

animate_heatmap()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(global_best[0]-0.2, global_best[0]+0.2, 100)
y = np.linspace(global_best[1]-0.2, global_best[1]+0.2, 100)
X_grid, Y_grid = np.meshgrid(x, y)
Z = f(X_grid, Y_grid)
ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D view of the function around global best')
ax.set_zlim(0, 0.4)
ax.scatter([global_best[0]], [global_best[1]], [global_best_score], color='red', s=50) 
plt.show()