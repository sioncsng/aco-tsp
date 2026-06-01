import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
random.seed(42)
np.random.seed(42)

# 1. GRAPH DEFINITION
NODE_NAMES = ['#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
N   = len(NODE_NAMES)
IDX = {n: i for i, n in enumerate(NODE_NAMES)}

INF = float('inf')
RAW = np.full((N, N), INF)
np.fill_diagonal(RAW, 0)

def add_edge(u, v, w):
    i, j = IDX[u], IDX[v]
    RAW[i][j] = RAW[j][i] = w

add_edge('#', 'A', 3)
add_edge('#', 'C', 2)
add_edge('#', 'G', 5)
add_edge('A', 'C', 6)
add_edge('B', 'C', 9)
add_edge('B', 'D', 8)
add_edge('C', 'F', 4)
add_edge('D', 'E', 7)
add_edge('D', 'H', 9)
add_edge('E', 'G', 1)
add_edge('E', 'H', 1)
add_edge('F', 'E', 2)
add_edge('G', 'H', 3)

# 2. PRECOMPUTE: Dijkstra ke # dan ke D
#    Fase 1: η = 1/(w + dist_to_#)  → terarah ke #
#    Fase 2: η = 1/(w + dist_to_D)  → terarah ke D
def dijkstra_to(target):
    dist = [INF] * N
    dist[target] = 0
    unvisited = set(range(N))
    while unvisited:
        u = min(unvisited, key=lambda x: dist[x])
        if dist[u] == INF:
            break
        unvisited.remove(u)
        for v in range(N):
            if RAW[u][v] not in (INF, 0):
                nd = dist[u] + RAW[u][v]
                if nd < dist[v]:
                    dist[v] = nd
    return dist

DIST_TO_HASH = dijkstra_to(IDX['#'])
DIST_TO_D    = dijkstra_to(IDX['D'])

# 3. ACO PARAMETERS (Eido & Ibrahim 2025)
ALPHA     = 1      # bobot pheromone
BETA      = 5      # bobot heuristic
RHO       = 0.1    # evaporation rate
Q         = 1.0    # konstanta deposit pheromone
N_ANTS    = 50     # jumlah semut
N_ITER    = 1000   # jumlah iterasi
TAU0      = 1.0    # pheromone awal
MAX_STEPS = 30     # batas langkah per semut

# 4. ACO CLASS
class AntColonyOptimizer:
    def __init__(self):
        self.tau = np.full((N, N), TAU0)

        # η fase 1: 1/(w + dist_to_#) — terarah ke #
        # η fase 2: 1/(w + dist_to_D) — terarah ke D
        self.eta_phase1 = np.zeros((N, N))
        self.eta_phase2 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if RAW[i][j] not in (INF, 0):
                    c1 = RAW[i][j] + DIST_TO_HASH[j]
                    c2 = RAW[i][j] + DIST_TO_D[j]
                    self.eta_phase1[i][j] = 1.0 / c1 if c1 > 0 else 0
                    self.eta_phase2[i][j] = 1.0 / c2 if c2 > 0 else 0

        self.best_path = None
        self.best_dist = INF
        self.history   = []

    def _neighbors(self, node):
        return [j for j in range(N) if RAW[node][j] not in (INF, 0)]

    def _build_path(self):
        """
        Semut berjalan dari H ke D, WAJIB lewat #.

        Sesuai paper Eido & Ibrahim (2025) — TSP klasik:
        - Tabu list: tiap node dikunjungi TEPAT SATU KALI
        - Fase 1 (H→#): η = 1/(w + dist_to_#) agar semut menuju #
        - Fase 2 (#→D): η = 1/(w + dist_to_D) agar semut menuju D
        - Formula: P(i→j) = [τ^α × η^β] / Σ[τ^α × η^β]
        """
        START = IDX['H']
        END   = IDX['D']
        HASH  = IDX['#']

        path      = [START]
        cur       = START
        hash_done = False
        dist      = 0.0

        # TABU LIST — no revisit
        tabu = {START}

        for _ in range(MAX_STEPS):
            if cur == END:
                break

            if cur == HASH and not hash_done:
                hash_done = True

            neighbors = self._neighbors(cur)

            if not hash_done:
                # Fase 1: blok D (belum lewat #), blok tabu
                candidates = [v for v in neighbors if v != END]
                filtered   = [v for v in candidates if v not in tabu]
            else:
                # Fase 2: blok tabu saja
                candidates = list(neighbors)
                filtered   = [v for v in candidates if v not in tabu]

            if not filtered:
                return None, INF

            # Formula probabilitas ACO (Eido & Ibrahim 2025)
            # P(i→j) = [τ(i,j)^α × η(i,j)^β] / Σ[τ(i,k)^α × η(i,k)^β]
            eta = self.eta_phase1 if not hash_done else self.eta_phase2

            scores = np.array([
                (self.tau[cur][v] ** ALPHA) * (eta[cur][v] ** BETA)
                for v in filtered
            ])
            total = scores.sum()
            probs = scores / total if total > 0 else np.ones(len(filtered)) / len(filtered)

            chosen = random.choices(filtered, weights=probs)[0]
            path.append(chosen)
            dist += RAW[cur][chosen]
            tabu.add(chosen)   # tambah ke tabu list
            cur = chosen

        # Validasi
        if path[-1] != END or HASH not in path:
            return None, INF
        return path, dist

    def _update_pheromone(self, solutions):
        """
        Pheromone update (Eido & Ibrahim 2025):
        τ(i,j) ← (1−ρ) × τ(i,j) + Σ Δτ(i,j)
        Δτ(i,j) = Q / L  (L = total cost path)
        """
        # Evaporation
        self.tau *= (1 - RHO)
        np.clip(self.tau, 1e-6, None, out=self.tau)

        # Deposit
        for path, dist in solutions:
            if path is None or dist == INF:
                continue
            deposit = Q / dist
            for k in range(len(path) - 1):
                i, j = path[k], path[k+1]
                self.tau[i][j] += deposit
                self.tau[j][i] += deposit

    def run(self):
        print("=" * 60)
        print("  ACO — H → (wajib #) → D")
        print("  Constraint : NO REVISIT (tabu list, TSP klasik)")
        print(f"  α={ALPHA}, β={BETA}, ρ={RHO}, Q={Q}")
        print(f"  Ants={N_ANTS}, Iters={N_ITER}, MaxSteps={MAX_STEPS}")
        print("=" * 60)

        for it in range(N_ITER):
            solutions = [self._build_path() for _ in range(N_ANTS)]
            valid = [(p, d) for p, d in solutions if p is not None and d < INF]

            if valid:
                best_it = min(valid, key=lambda x: x[1])
                if best_it[1] < self.best_dist:
                    self.best_dist = best_it[1]
                    self.best_path = best_it[0][:]

            self._update_pheromone(solutions)
            self.history.append(
                self.best_dist if self.best_dist < INF else None
            )

            if (it + 1) % 200 == 0:
                bd = f"{self.best_dist:.0f}" if self.best_dist < INF else "N/A"
                path_str = (
                    " → ".join(NODE_NAMES[i] for i in self.best_path)
                    if self.best_path else "none"
                )
                print(f"  Iter {it+1:4d}/{N_ITER}  |  best={bd}  |  {path_str}")

        return self.best_path, self.best_dist

# 5. VISUALIZATION
POS = {
    '#': (3.5, 4.0), 'A': (1.0, 3.0), 'B': (1.5, 1.0),
    'C': (3.5, 2.5), 'D': (5.0, 1.8), 'E': (7.0, 2.5),
    'F': (5.2, 3.2), 'G': (8.5, 3.5), 'H': (8.5, 1.0),
}

def draw_graph(ax, highlight_path=None, title=""):
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)

    # Semua edge
    for i in range(N):
        for j in range(i + 1, N):
            if RAW[i][j] < INF:
                u, v = NODE_NAMES[i], NODE_NAMES[j]
                x = [POS[u][0], POS[v][0]]
                y = [POS[u][1], POS[v][1]]
                ax.plot(x, y, color='#404060', lw=1.5, zorder=1)
                ax.text(
                    (x[0]+x[1])/2, (y[0]+y[1])/2,
                    str(int(RAW[i][j])),
                    color='#9999bb', fontsize=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1',
                              fc='#1a1a2e', ec='none')
                )

    # Highlight path
    if highlight_path:
        for k in range(len(highlight_path) - 1):
            ui, vi = highlight_path[k], highlight_path[k+1]
            u, v   = NODE_NAMES[ui], NODE_NAMES[vi]
            ax.plot([POS[u][0], POS[v][0]],
                    [POS[u][1], POS[v][1]],
                    color='#00ff88', lw=3.0, alpha=0.7, zorder=4)
            ax.annotate(
                "",
                xy=(POS[v][0], POS[v][1]),
                xytext=(POS[u][0], POS[u][1]),
                arrowprops=dict(arrowstyle="-|>", color='#00ff88',
                                lw=2.5, mutation_scale=20,
                                shrinkA=14, shrinkB=14),
                zorder=5
            )

    # Nodes
    for name, (x, y) in POS.items():
        if name == '#':   color, ec = '#c0392b', '#ff6b6b'
        elif name == 'H': color, ec = '#27ae60', '#2ecc71'
        elif name == 'D': color, ec = '#e67e22', '#ffd700'
        else:             color, ec = '#6c3483', '#a29bfe'
        ax.add_patch(plt.Circle((x, y), .30, color=color, zorder=4))
        ax.add_patch(plt.Circle((x, y), .30, color=ec,
                                fill=False, lw=2, zorder=5))
        ax.text(x, y, name, color='white', fontsize=9,
                fontweight='bold', ha='center', va='center', zorder=6)

    ax.set_xlim(-.3, 9.8)
    ax.set_ylim(.3, 4.8)
    ax.set_aspect('equal')
    ax.axis('off')

def visualize(aco):
    full_path  = aco.best_path
    total_dist = aco.best_dist
    path_names = [NODE_NAMES[i] for i in full_path]

    fig = plt.figure(figsize=(16, 9), facecolor='#0d0d1a')
    fig.suptitle(
        "TSP — Ant Colony Optimization  |  H → (wajib #) → D\n"
        f"α={ALPHA}, β={BETA}, ρ={RHO}, Q={Q}  |  "
        f"Ants={N_ANTS}, Iters={N_ITER}  |  No-Revisit (Tabu List)",
        color='white', fontsize=13, fontweight='bold', y=.99
    )

    # Panel 1: Graf asli
    ax1 = fig.add_subplot(2, 2, 1)
    draw_graph(ax1, title="Original Graph")
    ax1.legend(
        handles=[
            mpatches.Patch(color='#c0392b', label='# — mandatory node'),
            mpatches.Patch(color='#27ae60', label='H — start'),
            mpatches.Patch(color='#e67e22', label='D — end'),
            mpatches.Patch(color='#6c3483', label='other nodes'),
        ],
        facecolor='#1a1a2e', labelcolor='white',
        fontsize=8, loc='lower left'
    )

    # Panel 2: Best path
    ax2 = fig.add_subplot(2, 2, 2)
    draw_graph(ax2, highlight_path=full_path,
               title=f"Best Path  |  Total Cost = {total_dist:.0f}\n"
                     f"{' → '.join(path_names)}")
    ax2.legend(
        handles=[mpatches.Patch(color='#00ff88', label='best path')],
        facecolor='#1a1a2e', labelcolor='white',
        fontsize=8, loc='lower left'
    )

    # Panel 3: Convergence curve
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1a1a2e')
    valid_hist = [(i, v) for i, v in enumerate(aco.history) if v is not None]
    if valid_hist:
        xs, ys = zip(*valid_hist)
        ax3.plot(xs, ys, color='#00ff88', lw=2, label='Best Distance')
        ax3.fill_between(xs, ys, alpha=.15, color='#00ff88')
        min_dist = min(ys)
        min_idx  = ys.index(min_dist)
        ax3.scatter([xs[min_idx]], [min_dist],
                    color='#ffd700', s=80, zorder=5,
                    label=f'Optimal: {min_dist:.0f}')
        ax3.axhline(y=min_dist, color='#ffd700',
                    linestyle='--', alpha=0.5, lw=1)
    ax3.set_xlabel("Iteration", color='white')
    ax3.set_ylabel("Best Distance", color='white')
    ax3.set_title("Convergence Curve", color='white', fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    for sp in ax3.spines.values():
        sp.set_edgecolor('#555577')
    ax3.grid(True, color='#333355', alpha=.5)

    # Panel 4: Pheromone heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    tau_vis = aco.tau.copy()
    np.fill_diagonal(tau_vis, 0)
    tau_vis[RAW == INF] = 0
    im = ax4.imshow(tau_vis, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(N))
    ax4.set_xticklabels(NODE_NAMES, color='white', fontsize=9)
    ax4.set_yticks(range(N))
    ax4.set_yticklabels(NODE_NAMES, color='white', fontsize=9)
    ax4.set_title("Final Pheromone Matrix", color='white', fontweight='bold')
    plt.colorbar(im, ax=ax4)
    for i in range(N):
        for j in range(N):
            if tau_vis[i][j] > 0:
                ax4.text(j, i, f"{tau_vis[i][j]:.2f}",
                         ha='center', va='center',
                         fontsize=6, color='black')

    plt.tight_layout()
    out = "aco_tsp_result.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"\n[✓] Visualisasi disimpan → {out}")
    plt.close()

# 6. MAIN
def main():
    # Verifikasi manual path optimal no-revisit
    # H→G→#→C→F→E→D = 3+5+2+4+2+7 = 23
    print("=" * 60)
    print("  VERIFIKASI MANUAL PATH OPTIMAL")
    print("=" * 60)
    manual_path = ['H', 'G', '#', 'C', 'F', 'E', 'D']
    manual_cost = 0
    for k in range(len(manual_path) - 1):
        u, v = manual_path[k], manual_path[k+1]
        w = RAW[IDX[u]][IDX[v]]
        manual_cost += w
        print(f"    {u} → {v}  :  {w:.0f}")
    print(f"  {'─'*35}")
    print(f"  Total manual : {manual_cost:.0f}")
    print(f"  Expected     : 23")
    print(f"  Valid        : {'✓' if manual_cost == 23 else '✗'}")
    print()

    # Run ACO
    aco = AntColonyOptimizer()
    path, dist = aco.run()

    # Hasil final
    print("\n" + "=" * 60)
    print("  BEST RESULT")
    print("=" * 60)

    if path is None:
        print("  [!] Tidak ada path valid ditemukan.")
        return

    path_names = [NODE_NAMES[i] for i in path]
    print(f"  Path   : {' → '.join(path_names)}")
    print(f"  Cost   : {dist:.2f}")
    print(f"  Start H: {'✓' if path[0]  == IDX['H'] else '✗'}")
    print(f"  End D  : {'✓' if path[-1] == IDX['D'] else '✗'}")
    print(f"  Lewat #: {'✓' if IDX['#'] in path else '✗'}")
    print(f"  No Rev : {'✓' if len(path) == len(set(path)) else '✗'}")

    if abs(dist - 23) < 0.01:
        print("  [✓] OPTIMAL! H→G→#→C→F→E→D (cost=23)")
    else:
        print(f"  [!] Cost={dist:.0f}, expected 23")

    print("\n  Detail segment:")
    print(f"  {'─'*40}")
    total_check = 0
    for k in range(len(path) - 1):
        u = NODE_NAMES[path[k]]
        v = NODE_NAMES[path[k+1]]
        w = RAW[path[k]][path[k+1]]
        total_check += w
        print(f"    {u:>2} → {v:<2}  :  {w:.0f}")
    print(f"  {'─'*40}")
    print(f"    TOTAL :  {total_check:.0f}")

    visualize(aco)
    print("\n[✓] Selesai.")

if __name__ == "__main__":
    main()