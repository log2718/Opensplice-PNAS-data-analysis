import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "arush_data" / "exonic_variants.csv")

''' for test trials
selected_exons = [
    "ATM_e56", #stubborn
    "BLM_e17", #stubborn
    "PHYHD1_e11",
    "NPC1L1_e7",
    "GDPD2_e2",
    "ADGRG3_e3",
    "ABLIM3_e14", #2 clusters
    "TRPM6_e23",
    "SNRNP70_e8",
    "SCN5A_e6",
    "SCN8A_e6",
]
'''

#df = df[df["exon_id"].isin(selected_exons)]
df = df.dropna(subset=['psi', 'logit_psi']) #has to have values for psi and the logit

df.to_csv(ROOT / "outputs" / "test_exons.csv", index=False)

print(f'Variants: {len(df)}')
print(f'Exons: {df['exon_id'].nunique()}')
