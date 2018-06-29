import pandas as pd
data = pd.read_csv('/Users/harsh/Desktop/QUINT_F/csv_files/quint_tags.csv')
data.columns = ['content_tag','views']
b = data['content_tag'].value_counts().to_dict()
c = pd.DataFrame(b.items(), columns=['content_tag', 'number'])
#print c
f['sum'] = f.groupby(['content_tag'])['views'].transform('sum')
f = pd.merge(f,c, how='left', on = 'content_tag')
del f['views']
#print f
f = f.drop_duplicates()
#print f
f['avg'] = f['sum']/f['number']
#print f
e = f['avg'].sum()
f['content_tag_score'] = f['avg']/e
del f['sum']
del f['number']
del f['avg']
f.to_csv('/Users/harsh/Desktop/QUINT_F/csv_files/quint_tag_score.csv')