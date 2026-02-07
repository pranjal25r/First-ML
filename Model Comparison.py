import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)

plt.scatter(x=y_train, y = y_lr_train_pred, c = "#7CAE00", alpha = 0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train,p(y_train),'#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
