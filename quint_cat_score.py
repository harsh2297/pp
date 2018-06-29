import pandas as pd
f = pd.read_csv('/Users/harsh/Desktop/QUINT_F/csv_files/Quint_category.csv')
f.columns = ['content_category','views']
b = f['content_category'].value_counts().to_dict()
c = pd.DataFrame(b.items(), columns=['content_category', 'number'])
#print c
f['sum'] = f.groupby(['content_category'])['views'].transform('sum')
f = pd.merge(f,c, how='left', on = 'content_category')
del f['views']
print f
f = f.drop_duplicates()
#print f
f['avg'] = f['sum']/f['number']
#print f
e = f['avg'].sum()
f['content_category_score'] = f['avg']/e
del f['sum']
del f['number']
del f['avg']
f.to_csv('/Users/harsh/Desktop/QUINT_F/csv_files/Quint_cat_score.csv')

