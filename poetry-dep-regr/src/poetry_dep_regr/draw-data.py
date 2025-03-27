import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import statistics
import scipy
from functools import cache
from sklearn.linear_model import LinearRegression 
from adjustText import adjust_text
from poetry_dep_regr.data import df
from poetry_dep_regr.basest import basest
# --- CHART ---
fig, ax = plt.subplots()
ax.scatter(df["in_queue"], df["received"], marker='o', c="red")
ax.set_xlabel("number of families in the queue")
ax.set_ylabel("number of families who have received housing")   

#--- SIGNATURES ---
c = 1
сс = 1
for x, y, l in zip(df["in_queue"], df["received"], df["year"]):
    ax.annotate(l, xy=(x,y), textcoords="offset points", xytext=(1.8*(-c*2.2), 80*сс), xycoords='data', 
                arrowprops=dict(arrowstyle='->', color='blue', lw=0.2, ls='-', 
                connectionstyle='angle3'))
    c += 1.1
    сс *= (-1)

#--- STATISTICS ---

bst = basest(df["in_queue"], df["received"])
plt.plot(df["in_queue"], bst.y_pred, color="red", label='Regression line')
# Regression line

for i in range(len(df["in_queue"])):
    plt.plot([df["in_queue"][i], df["in_queue"][i]], 
    [df["received"][i], bst.y_pred[i]], 
    color='green', label='Disp line')
# Disp line

bst.print_summary()
# print 

# --- ГИСТОГРАММА ОСТАТКОВ ---
fig2, ax2 = plt.subplots()
residuals = bst.y - bst.y_pred
ax2.hist(residuals, bins='auto', density=True, color='blue', alpha=0.7)
ax2.set_title("Гистограмма остатков")
ax2.set_xlabel("Остатки")
ax2.set_ylabel("Частота")

# --- RENDERING ---
plt.show() 