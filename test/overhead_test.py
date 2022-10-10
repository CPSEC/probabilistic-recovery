import csv
from statistics import mean
import pandas as pd
file = r'../rtss/res/tank_DIFF_K_overhead.csv'
with open(file, newline='') as csvfile:
    rst = {}
    for i in range(1,16):
        rst[str(i)] = []
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        rst[row[0]].append(float(row[1]))

for i in range(1,16):
    print(i, mean(rst[str(i)]))
#         rst.append(row)
# rst = pd.read_csv(file, header=None)
# rst = rst.to_numpy()
# sum =
# for i in len(rst):
