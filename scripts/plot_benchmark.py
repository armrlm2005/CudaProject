"""
Plot CUDA SGEMM benchmark results: GFLOPS vs matrix size, one line per
kernel, with cuBLAS as the reference ceiling.

Usage:
    python plot_benchmark.py

Edit the `data` dict below with your own numbers from
`scripts/run_benchmark.sh` output (already filled in with the T4 results
from this conversation - replace with your own if you re-run it).

Requires: matplotlib  (pip install matplotlib --break-system-packages)
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Benchmark data: {kernel_name: {size: gflops}}
# Replace these with your own numbers if you re-run the benchmark.
# ---------------------------------------------------------------------
sizes = [128, 256, 512, 1024, 2048, 4096]

data = {
    "0: cuBLAS (reference)": [327.2, 1273.3, 4034.4, 5003.8, 4749.0, 4214.9],
    "1: Naive":               [26.8,   37.3,   42.3,   57.9,   61.8,   61.8],
    "2: GMEM Coalescing":     [188.0, 500.5,  727.8,  709.8,  573.0,  518.1],
    "3: SMEM Tiling":         [291.5, 595.9,  931.7, 1057.2,  905.4,  836.9],
    "4: 1D Blocktiling":      [174.6, 774.7, 1770.0, 2045.2, 1834.0, 1621.7],
    "5: 2D Blocktiling":      [72.1,  328.7, 1809.1, 2749.3, 3505.7, 2806.2],
    "6: Vectorized":          [95.5,  403.5, 1735.2, 2961.0, 3797.5, 2990.4],
    "7: Warptiling":          [90.9,  402.6, 1752.6, 3354.1, 4833.4, 3905.9],
}

# Distinct colors per kernel; cuBLAS reference drawn dashed + black so it
# always reads as "the ceiling" rather than just another series.
colors = {
    "0: cuBLAS (reference)": "#000000",
    "1: Naive":              "#d62728",
    "2: GMEM Coalescing":    "#ff7f0e",
    "3: SMEM Tiling":        "#bcbd22",
    "4: 1D Blocktiling":     "#2ca02c",
    "5: 2D Blocktiling":     "#17becf",
    "6: Vectorized":         "#1f77b4",
    "7: Warptiling":         "#9467bd",
}

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

for label, values in data.items():
    is_reference = label.startswith("0:")
    ax.plot(
        sizes,
        values,
        marker="o",
        markersize=5,
        linewidth=2.5 if is_reference else 2,
        linestyle="--" if is_reference else "-",
        color=colors[label],
        label=label,
        zorder=3 if is_reference else 2,
    )

ax.set_xscale("log", base=2)
ax.set_xticks(sizes)
ax.set_xticklabels([str(s) for s in sizes])

ax.set_xlabel("Matrix size (M = N = K)", fontsize=11)
ax.set_ylabel("Throughput (GFLOPS)", fontsize=11)
ax.set_title(
    "CUDA SGEMM Kernel Progression vs cuBLAS\n(NVIDIA T4, FP32, 20 iterations averaged)",
    fontsize=13,
    fontweight="bold",
)

ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig("gemm_benchmark.png", dpi=200, bbox_inches="tight")
print("Saved chart to gemm_benchmark.png")

plt.show()
