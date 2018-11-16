import pandas as pd
from snownlp import SnowNLP, sentiment

sentiment.train('E:\\data\\low.txt','E:\\data\\high.txt')
mix=pd.read_csv('E:\\data\\mix.csv',encoding='gbk')
mix=mix.dropna()
right=0
wrong=0
for i in mix.index:
    txt=mix.loc[i,'txt']
    fen=mix.loc[i,'fen']
    s = SnowNLP(txt)
    f=s.sentiments
    print((f,fen))
    if  f>0.5 and fen>2.5 or f<0.5 and fen<2.5:
        right += 1
    else:
        wrong += 1
print(right/(right+wrong))
