import pandas as pd

all_paths = []
all_qscores = []

with open('qscore.txt') as f:
    for line in f.readlines():
        path = line.split('\t')[0]
        score = float(line.split('\t')[1].split('\n')[0])
        if score >= 0.1:
            all_paths.append(path)
            all_qscores.append(score)


df = pd.DataFrame()


df['path'] = all_paths
df['qscore'] = all_qscores

df.to_csv('data.csv')