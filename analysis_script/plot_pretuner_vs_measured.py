from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import linregress, spearmanr

#Paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT / "outputs" / "test_exons_with_pretuner.csv"
OUTPUT_DIR = ROOT / "outputs" / "plots"

#Main
def main():
    df = pd.read_csv(INPUT_CSV) 
    for exon_id, group in df.groupby("exon_id"):
        fig, axes = plt.subplots(
            1,
            2,
            figsize = (12,5)
        )

        #logitPSI vs pretuner PNAS pred
        g = group.dropna(
            subset=["pretuner", "logit_psi"]
        )
        x = g["pretuner"].to_numpy()
        y = g["logit_psi"].to_numpy()
        slope, intercept, r, p, stderr = linregress(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        axes[0].scatter(
            x,
            y,
            alpha=0.5, #Transparency
            s=8, #marker area
        )

        # regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line
        axes[0].plot(
            x_line,
            y_line,
        )
        axes[0].set_xlabel("PNAS pretuner output")
        axes[0].set_ylabel("Measured logit PSI")
        axes[0].set_title(
            f"n = {len(g)}\n"
            f"Slope = {slope:.3f}, Pearson r = {r:.3f}\n"
            f"Spearman ρ = {spearman_r:.3f}"
        )

        # PSI against pretuner for 3 replicates
        for col, label, color in [
            ("psi_r1", "R1", "red"),
            ("psi_r2", "R2", "green"),
            ("psi_r3", "R3", "blue"),
        ]:
            rgroup = group.dropna(
                subset=["pretuner", col]
            )

            x_rep = rgroup["pretuner"].to_numpy()
            y_rep = rgroup[col].to_numpy()

            slope_rep, intercept_rep, r_rep, p_rep, stderr_rep = linregress(
                x_rep,
                y_rep
            )
            spearman_rep, spearman_p_rep = spearmanr(
                x_rep,
                y_rep
            )
            axes[1].scatter(
                x_rep,
                y_rep,
                alpha=0.45,
                s=7,
                color=color,
                label=(
                    f"{label}: "
                    f"slope={slope_rep:.3f}, "
                    f"r={r_rep:.3f}, "
                    f"ρ={spearman_rep:.3f}"
                ),
            )
        axes[1].set_xlabel("PNAS pretuner output")
        axes[1].set_ylabel("Raw replicate PSI")
        axes[1].legend()
        fig.suptitle(exon_id)
        plt.tight_layout()
        
        out = OUTPUT_DIR / f"{exon_id}_with_ss_pretuner_vs_measured.png"
        plt.savefig(
            out,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved {out}")

if __name__ == "__main__":
    main()