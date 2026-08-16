import pandas as pd

df = pd.read_csv('./arush_data/exonic_variants.csv')
print(df.columns)

print(df['lib_id'].value_counts())
regions = ['region', 'region_start', 'region_end']
for r in regions:
    print(df[r].value_counts())


exons_multiple_libs = df.groupby('ensembl_exon_id')['lib_id'].nunique()
exons_multiple_libs = exons_multiple_libs[exons_multiple_libs > 1]
print(exons_multiple_libs)

print(df[df['ensembl_exon_id'].isin(exons_multiple_libs.index)])
print()
#see ABLIM3_e14


abl = df[df["exon_id"] == "ABLIM3_e14"]

print("Total:", len(abl))

print("psi missing:")
print(abl["psi"].isna().value_counts())

print("\nlogit_psi missing:")
print(abl["logit_psi"].isna().value_counts())

print("\nBoth available:")
print(
    abl[["psi", "logit_psi"]]
    .notna()
    .all(axis=1)
    .value_counts()
)

# after selecting exons
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
df_selected = df[df["exon_id"].isin(selected_exons)]

print(
    "After exon selection:",
    (df_selected["exon_id"] == "ABLIM3_e14").sum()
)

# after dropna
df_filtered = df_selected.dropna(
    subset=["psi", "logit_psi"]
)

print(
    "After dropna:",
    (df_filtered["exon_id"] == "ABLIM3_e14").sum()
)