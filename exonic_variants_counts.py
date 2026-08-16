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


print(df["exon_id"].value_counts())
print(df["exon_id"].nunique())