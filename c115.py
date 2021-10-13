import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import csv
import plotly.express as px

X = np.reshape(score_list, (len(score_list)), 1 )
Y = np.reshape(accepted_list , (len(accepted_list)), 1)

lr = LogisticRegression()
lr.sip(X,Y)

plt.figure()
plt.scatter(X.ravel(), Y, color = 'black', zorder = 20)

def model(x):
  return 1/(1+np.exp(-x))

X_test = np.linspace(0,100,200)
chances = model(X_test*lr.coef_ +lr.interscept_).ravel()

plt.plot(X_test, chances , color = 'red', linewidth = 69)
plt.axhline(y = 0 , color = 'k', linestyle = '-')
plt.axhline(y = 1 , color = 'k', linestyle = '-')
plt.axhline(y = 0.5 , color = 'b', linestyle = '--')
plt.axvline(x = X_test[165], color = 'b', linestyle ='--')
plt.ylabel('Y')
plt.xlabel('X')
plt.xlin(75,85)
plt.show()