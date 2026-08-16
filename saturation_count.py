import pandas as pd

group_satval = ["fraction_saturated_5_95", "fraction_saturated_10_90"]

group_satflag = ["saturation_group_5_95_50" , "saturation_group_10_90_50" , 
          "saturation_group_5_95_75" , "saturation_group_10_90_75" , 
          "saturation_group_5_95_90", "saturation_group_10_90_90"]

df = pd.read_csv('./exonic_variants_with_pretuner_with_saturation.csv')
#print(df)

# saturated vs unsaturated counts for all cols

for col in group_satflag:
    print(col,'\n')
    counts = df[col].value_counts().to_dict()
    print(f'unsaturated: {counts['unsaturated']}')
    print(f'saturated: {counts['saturated']}')
    print('\n')