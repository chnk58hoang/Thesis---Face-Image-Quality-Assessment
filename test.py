import pandas as pd

df = pd.read_csv('data.csv')

l1 = df[df['br']==0]
l2 = df[df['br']==1]

print(len(l1))
print(len(l2))