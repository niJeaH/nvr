import pandas as pd
import numpy as np
import itertools

path1 = "C:/Users/HaeJinKim/Desktop/2018-2019/Paper/ACL/AMR/Amazone crawling data/"
xiaomi_rev=pd.read_excel(path1+'Xiaomi_raw.xlsx')

xiaomi_rev.columns
revs = [d for d in xiaomi_rev.review if type(d) != type(0.1)]
raw_txt = itertools.chain.from_iterable(revs)

raw_txt = "<eod> ".join(revs)
raw_txt
f = open('rawtxt.txt','w+')
f.write(raw_txt)
f.close()