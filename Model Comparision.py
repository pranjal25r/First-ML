import pandas as pd
import matplotlib.pyplot as plt

df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)

