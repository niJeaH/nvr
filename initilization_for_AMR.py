print('hello')

import numpy as np
import pandas as pd
import torch as tc
import sys
sys.path.insert(0, 'C:/Users/HaeJinKim/Desktop/2018-2019/macros/Python')
import semantic_summ_master.src.fei as amr
import jamr_Semeval_2016 as jamr_2016
import csv
'''C:/Users/HaeJinKim/Desktop/2018-2019/Paper/ACL/AMR/Amazone crawling data/'''
def rev   

rev = pd.read_csv("C:/Users/HaeJinKim/Desktop/2018-2019/Paper/ACL/AMR/Amazone crawling data/review_all.csv", header= 0, encoding="iso-8859-1")
rev.head 

rev.lookup(1,1)