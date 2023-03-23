import pandas as pd

csv = '/home/artorias/PycharmProjects/weights/new_data.csv'
train_val_dataframe = pd.read_csv(csv).iloc[:102400, :]
test_df = pd.read_csv(csv).iloc[102400:, :]

for i in range(7):
    l = test_df[test_df['pose'] == i]
    print(len(l))
    print(len(l) * 100/len(test_df))