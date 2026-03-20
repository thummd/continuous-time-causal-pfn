"""
Plot the 6 TSCM Sampler causal structures for identifiability case studies.

Uses the same visual style as ctp/plot_causal_graphs.py for consistency.
Generates a 2x3 grid showing all temporal DAGs with instantaneous,
lagged, and autoregressive edges.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style config (matches ctp/plot_causal_graphs.py)
NODE_SIZE = 1800
FONT_SIZE = 13
EDGE_COLOR_INSTANT = "#2C3E50"
EDGE_COLOR_LAGGED = "#E74C3C"
EDGE_COLOR_AUTO = "#7F8C8D"
NODE_COLOR = "#ECF0F1"
NODE_COLOR_HIDDEN = "#FADBD8"  # light red for unobserved variables
NODE_EDGE_COLOR = "#2C3E50"
NODE_EDGE_COLOR_HIDDEN = "#E74C3C"  # red outline for unobserved
ARROW_STYLE = "Simple,tail_width=1.2,head_width=8,head_length=6"
FIG_BG = "white"


def draw_temporal_dag(ax, nodes_t0, nodes_t1, edges, title,
                      hidden_indices=None,
                      x_offset_t0=0, x_offset_t1=3.5, y_spacing=1.8):
    """
    Draw a temporal causal DAG with two time slices.

    Parameters
    ----------
    nodes_t0 : list of str - node labels at t-1
    nodes_t1 : list of str - node labels at t
    edges : list of (src_slice, src_idx, dst_slice, dst_idx, style)
            slice: 0 = t-1, 1 = t
            style: 'instant', 'lagged', 'auto'
    hidden_indices : set of int - indices of hidden/unobserved nodes
    """
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    hidden_indices = hidden_indices or set()

    # Compute positions
    pos = {}
    max_nodes = max(len(nodes_t0), len(nodes_t1))

    for i, label in enumerate(nodes_t0):
        y = (max_nodes - 1) / 2 * y_spacing - i * y_spacing
        pos[(0, i)] = (x_offset_t0, y)

    for i, label in enumerate(nodes_t1):
        y = (max_nodes - 1) / 2 * y_spacing - i * y_spacing
        pos[(1, i)] = (x_offset_t1, y)

    # Draw time slice labels
    all_y = [p[1] for p in pos.values()]
    y_top = max(all_y) + 1.2
    ax.text(x_offset_t0, y_top, r"$t-1$", ha="center", fontsize=14,
            fontstyle="italic", color="#555555")
    ax.text(x_offset_t1, y_top, r"$t$", ha="center", fontsize=14,
            fontstyle="italic", color="#555555")

    # Draw dashed time separator
    x_mid = (x_offset_t0 + x_offset_t1) / 2
    ax.axvline(x_mid, color="#CCCCCC", linestyle="--", linewidth=1, zorder=0)

    # Draw edges
    style_map = {
        "instant": (EDGE_COLOR_INSTANT, "-", 2.0),
        "lagged": (EDGE_COLOR_LAGGED, "-", 2.0),
        "auto": (EDGE_COLOR_AUTO, "--", 1.5),
    }

    for src_slice, src_idx, dst_slice, dst_idx, style in edges:
        src_pos = pos[(src_slice, src_idx)]
        dst_pos = pos[(dst_slice, dst_idx)]
        color, ls, lw = style_map[style]

        dx = dst_pos[0] - src_pos[0]
        dy = dst_pos[1] - src_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0:
            continue
        shrink = 0.35
        sx = src_pos[0] + shrink * dx / dist
        sy = src_pos[1] + shrink * dy / dist
        ex = dst_pos[0] - shrink * dx / dist
        ey = dst_pos[1] - shrink * dy / dist

        # Use slight curve for edges within same time slice to avoid overlap
        if src_slice == dst_slice and abs(src_idx - dst_idx) > 1:
            rad = 0.2
        else:
            rad = 0.0

        arrow = mpatches.FancyArrowPatch(
            (sx, sy), (ex, ey),
            arrowstyle=ARROW_STYLE,
            color=color, linestyle=ls, linewidth=lw,
            mutation_scale=1, zorder=1,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

    # Draw nodes
    for slice_idx, node_list in [(0, nodes_t0), (1, nodes_t1)]:
        for i, label in enumerate(node_list):
            x, y = pos[(slice_idx, i)]
            is_hidden = i in hidden_indices
            fc = NODE_COLOR_HIDDEN if is_hidden else NODE_COLOR
            ec = NODE_EDGE_COLOR_HIDDEN if is_hidden else NODE_EDGE_COLOR
            ls = "--" if is_hidden else "-"

            circle = plt.Circle((x, y), 0.3, facecolor=fc,
                                edgecolor=ec, linewidth=2,
                                linestyle=ls, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=FONT_SIZE, fontweight="bold", zorder=3)

    # Formatting
    all_x = [p[0] for p in pos.values()]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1.2, y_top + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def add_legend(fig):
    """Add a shared legend for edge types and node types."""
    legend_elements = [
        mpatches.Patch(facecolor=EDGE_COLOR_INSTANT, label="Instantaneous edge"),
        mpatches.Patch(facecolor=EDGE_COLOR_LAGGED, label="Lagged edge"),
        mpatches.Patch(facecolor=EDGE_COLOR_AUTO, label="Autoregressive"),
        mpatches.Patch(facecolor=NODE_COLOR_HIDDEN, edgecolor=NODE_EDGE_COLOR_HIDDEN,
                       linestyle="--", label="Unobserved variable"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=11, frameon=True, fancybox=True,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.01))


def add_identification_note(ax, text):
    """Add identification strategy note below the DAG."""
    ax.text(0.5, -0.08, text, transform=ax.transAxes,
            ha="center", fontsize=9.5, family="serif",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA",
                      edgecolor="#DEE2E6"))


def plot_tscm_structures():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("TSCM Sampler: Identifiability Case Study Structures",
                 fontsize=17, fontweight="bold", y=0.97)

    # --- (a) Observed Confounder: Z -> X (instant), Z(t-1) -> Y(t) (lagged) ---
    # Nodes: Z, X, Y. All autoregressive.
    draw_temporal_dag(
        axes[0, 0],
        nodes_t0=["Z", "X", "Y"],
        nodes_t1=["Z", "X", "Y"],
        edges=[
            (0, 0, 1, 0, "auto"),    # Z(t-1) -> Z(t)
            (0, 1, 1, 1, "auto"),    # X(t-1) -> X(t)
            (0, 2, 1, 2, "auto"),    # Y(t-1) -> Y(t)
            (1, 0, 1, 1, "instant"), # Z(t) -> X(t)
            (0, 0, 1, 2, "lagged"),  # Z(t-1) -> Y(t)
        ],
        title="(a) Observed Confounder",
    )
    add_identification_note(axes[0, 0],
        r"$p(y_t|do(x_t), H_{t-1}) = \int p(y_t|x_t, z_t)\, p(z_t|z_{t-1})\, dz_t$"
        "\nBack-door adjustment through Z")

    # --- (b) Mediator: X(t-1) -> M(t) (lagged), M(t) -> Y(t) (instant) ---
    # Nodes: X, M, Y. All autoregressive.
    draw_temporal_dag(
        axes[0, 1],
        nodes_t0=["X", "M", "Y"],
        nodes_t1=["X", "M", "Y"],
        edges=[
            (0, 0, 1, 0, "auto"),    # X(t-1) -> X(t)
            (0, 1, 1, 1, "auto"),    # M(t-1) -> M(t)
            (0, 2, 1, 2, "auto"),    # Y(t-1) -> Y(t)
            (0, 0, 1, 1, "lagged"),  # X(t-1) -> M(t)
            (1, 1, 1, 2, "instant"), # M(t) -> Y(t)
        ],
        title="(b) Mediator",
    )
    add_identification_note(axes[0, 1],
        r"$p(y_t|do(x_t)) = \sum_m p(y_t|m_t)\, p(m_t|x_{t-1})$"
        "\nCausal effect fully mediated through M")

    # --- (c) Confounder + Mediator: Z->X->M->Y (instant), Z(t-1)->Y(t) (lagged) ---
    # Nodes: Z, X, M, Y. All autoregressive.
    draw_temporal_dag(
        axes[0, 2],
        nodes_t0=["Z", "X", "M", "Y"],
        nodes_t1=["Z", "X", "M", "Y"],
        edges=[
            (0, 0, 1, 0, "auto"),    # Z(t-1) -> Z(t)
            (0, 1, 1, 1, "auto"),    # X(t-1) -> X(t)
            (0, 2, 1, 2, "auto"),    # M(t-1) -> M(t)
            (0, 3, 1, 3, "auto"),    # Y(t-1) -> Y(t)
            (1, 0, 1, 1, "instant"), # Z(t) -> X(t)
            (1, 1, 1, 2, "instant"), # X(t) -> M(t)
            (1, 2, 1, 3, "instant"), # M(t) -> Y(t)
            (0, 0, 1, 3, "lagged"),  # Z(t-1) -> Y(t)
        ],
        title="(c) Confounder + Mediator",
    )
    add_identification_note(axes[0, 2],
        "Hybrid: adjust for Z + trace through M\n"
        r"$Z \to X \to M \to Y$, $Z(t\!-\!1) \to Y(t)$")

    # --- (d) Unobserved Confounder: U->X, U->Y (instant), U hidden ---
    # Nodes: U, X, Y. All autoregressive.
    draw_temporal_dag(
        axes[1, 0],
        nodes_t0=["U", "X", "Y"],
        nodes_t1=["U", "X", "Y"],
        edges=[
            (0, 0, 1, 0, "auto"),    # U(t-1) -> U(t)
            (0, 1, 1, 1, "auto"),    # X(t-1) -> X(t)
            (0, 2, 1, 2, "auto"),    # Y(t-1) -> Y(t)
            (1, 0, 1, 1, "instant"), # U(t) -> X(t)
            (1, 0, 1, 2, "instant"), # U(t) -> Y(t)
        ],
        title="(d) Unobserved Confounder",
        hidden_indices={0},
    )
    add_identification_note(axes[1, 0],
        "U confounds X and Y but is unobserved\n"
        "Not point-identifiable without instrument/mediator")

    # --- (e) Back-Door: Z->X, Z->Y, X->Y (all instant) ---
    # Nodes: Z, X, Y. All autoregressive.
    draw_temporal_dag(
        axes[1, 1],
        nodes_t0=["Z", "X", "Y"],
        nodes_t1=["Z", "X", "Y"],
        edges=[
            (0, 0, 1, 0, "auto"),    # Z(t-1) -> Z(t)
            (0, 1, 1, 1, "auto"),    # X(t-1) -> X(t)
            (0, 2, 1, 2, "auto"),    # Y(t-1) -> Y(t)
            (1, 0, 1, 1, "instant"), # Z(t) -> X(t)
            (1, 0, 1, 2, "instant"), # Z(t) -> Y(t)
            (1, 1, 1, 2, "instant"), # X(t) -> Y(t)
        ],
        title="(e) Back-Door",
    )
    add_identification_note(axes[1, 1],
        r"$p(y_t|do(x_t), H_{t-1}) = \sum_z p(y_t|x_t, z_t)\, p(z_t|H_{t-1})$"
        "\nZ satisfies the back-door criterion")

    # --- (f) Front-Door: X->M->Y (instant), U->X, U->Y (instant), U hidden ---
    # Nodes: U, X, M, Y. All autoregressive.
    draw_temporal_dag(
        axes[1, 2],
        nodes_t0=["U", "X", "M", "Y"],
        nodes_t1=["U", "X", "M", "Y"],
        edges=[
            (0, 0, 1, 0, "auto"),    # U(t-1) -> U(t)
            (0, 1, 1, 1, "auto"),    # X(t-1) -> X(t)
            (0, 2, 1, 2, "auto"),    # M(t-1) -> M(t)
            (0, 3, 1, 3, "auto"),    # Y(t-1) -> Y(t)
            (1, 0, 1, 1, "instant"), # U(t) -> X(t)
            (1, 0, 1, 3, "instant"), # U(t) -> Y(t)
            (1, 1, 1, 2, "instant"), # X(t) -> M(t)
            (1, 2, 1, 3, "instant"), # M(t) -> Y(t)
        ],
        title="(f) Front-Door",
        hidden_indices={0},
    )
    add_identification_note(axes[1, 2],
        r"$p(y|do(x)) = \sum_m p(m|x) \sum_{x'} p(y|x',m)\, p(x')$"
        "\nFront-door criterion through mediator M")

    add_legend(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # Save
    fig.savefig("figures/tscm_structures.png", dpi=300,
                bbox_inches="tight", facecolor=FIG_BG)
    fig.savefig("figures/tscm_structures.pdf",
                bbox_inches="tight", facecolor=FIG_BG)
    print("Saved to figures/tscm_structures.png and .pdf")
    plt.close()


if __name__ == "__main__":
    plot_tscm_structures()
