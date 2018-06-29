import pandas as pd

data1 = pd.read_csv("/Users/harsh/Desktop/QUINT_F/csv_files/text_processed.csv", header  = None)
data1.columns = ['id','category','text']

data2 = pd.read_csv('/Users/harsh/Desktop/pos/fastnextscoring/cldf.csv', header = None)
data2.columns = ['cluster','text']

data1 = pd.merge(data1,data2, how = 'left', on = 'text')
data1.to_csv('processed_text_with_clustering.csv')