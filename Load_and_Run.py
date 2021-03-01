pip3 install numpy
pip3 install pandas
pip3 install networkx
pip3 install matplotlib
pip3 install datetime
#! /urs/bin/python
bash
cd jamr
python3

import pexpect
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import time, sys, os, re
import sentence_splitter

def Filter(string, substr): 
    return [str for str in string if
             any(sub in str for sub in substr)] 

file_all = "/mnt/c/Users/HaeJinKim/Desktop/2018-2019/Paper/ACL/AMR/\
Amazone crawling data/all_rev_2.csv"
revs_all = pd.read_csv(file_all,encoding = 'unicode_escape')
revs_all = revs_all.iloc[:,1:12]
revs_all.columns

dt0 = revs_all.get("rvdate").apply(lambda x : datetime.strptime(x,'%B %d, %Y') \
    if type(x)==type("text") else x)
revs_all["rv_date"] = dt0
del dt0

year = []
for y in [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
    for m in range(12):
        year.append(str(y)+"-"+str(m+1))

Brand_list=revs_all.brand.unique()

df = pd.DataFrame({"date":year})
for brd in Brand_list:
    revs_br = revs_all[revs_all.brand==brd]
    brand_value = pd.DataFrame({str(brd):[]})
    for i in [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
        year = i
        for j in range(12):
            upper = j+2
            lower = j+1
            year_a = year
            if upper == 13:
                year_a += 1
                upper = 1
            upper_b = revs_br['rv_date']<datetime.strptime(str(year_a)+"-"+str(upper),"%Y-%m")
            lower_b = revs_br['rv_date']>=datetime.strptime(str(year)+"-"+str(lower),"%Y-%m")
            brand_value = brand_value.append({str(brd):len(revs_br[upper_b&lower_b])},ignore_index= True)
    df.insert(1,brd,brand_value,True)


### ### sort by brand
### load the file of a brand LG
file_LG = "/mnt/c/Users/HaeJinKim/Desktop/2018-2019/Paper/ACL/AMR/\
Amazone crawling data/LG_raw.csv"
revs_LG = pd.read_csv(file_LG,encoding = 'unicode_escape')
revs_LG = revs_LG.iloc[:,2:12]
revs_LG.columns


### ### ### sort by month
############Extract review of year by month
for i in range(12):
    upper = i+2
    lower = i+1
    year_a = year
    if upper == 13:
        year_a += 1
        upper = 1
    upper_b = revs_LG['rv_date']<datetime.strptime(str(year_a)+"-"+str(upper),"%Y-%m")
    lower_b = revs_LG['rv_date']>=datetime.strptime(str(year)+"-"+str(lower),"%Y-%m")
    revs_LG[upper_b&lower_b]

value_LG = []
value_Hw = []
value_Sa = []
value_Ap = []
for i in [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
    year = i
    for j in range(12):
        upper = j+2
        lower = j+1
        year_a = year
        if upper == 13:
            year_a += 1
            upper = 1
        upper_b = revs_LG['rv_date']<datetime.strptime(str(year_a)+"-"+str(upper),"%Y-%m")
        lower_b = revs_LG['rv_date']>=datetime.strptime(str(year)+"-"+str(lower),"%Y-%m")
        value_LG.append(len(revs_LG[upper_b&lower_b]))

desc = pd.DataFrame(data = {"year": year, "hw" : value_Hw,\
    "sa" : value_Sa, "lg" :value_LG, "ap":value_Ap})

################################################################72limit79limit
### ### ### ### run the AMR per each review
revs_Hw=revs_all[revs_all.brand==Brand_list[0]]
revs_LG=revs_all[revs_all.brand==Brand_list[4]]

Pname_LG=revs_LG.Product_name.unique()
Pname_Hw=revs_Hw.Product_name.unique()
len(Pname_LG)
len(Pname_Hw)
#flagship_Hw = ['Mate 9','Mate 10', 'P9', 'P10']
#flagship_LG = ['V20','V30','V30+','G5','G6']
#flagship_Hw = ['P8','P9', 'P10']
#flagship_LG = ['G5','G6']
flagship_Hw = ['Mate 8','Mate 9','Mate 10']
flagship_LG = ['V10','V20','V30','V30+']

flagship_Hw = ['P8','P9','P10','Ascend']
flagship_LG = ['G4','G5','G6']
revs_LG.Product_name[(revs_LG["rv_date"]<datetime.strptime('2018-01',"%Y-%m"))&\
    (revs_LG["rv_date"]>=datetime.strptime('2017-01',"%Y-%m"))].unique()
revs_Hw.Product_name[(revs_Hw["rv_date"]<datetime.strptime('2018-01',"%Y-%m"))&\
    (revs_Hw["rv_date"]>=datetime.strptime('2017-01',"%Y-%m"))].unique()

target_LG=revs_LG[revs_LG.Product_name.isin(Filter(Pname_LG,flagship_LG))]
target_Hw=revs_Hw[revs_Hw.Product_name.isin(Filter(Pname_Hw, flagship_Hw))]

subtarget_LG=target_LG[target_LG["rv_date"]<datetime.strptime('2018-01',"%Y-%m")]
subtarget_Hw=target_Hw[target_Hw["rv_date"]<datetime.strptime('2018-01',"%Y-%m")]

batch1_LG = target_LG[target_LG["rv_date"]<datetime.strptime('2017-02',"%Y-%m")]
batch1_HW = target_Hw[target_Hw["rv_date"]<datetime.strptime('2017-02',"%Y-%m")]
batch2_LG = subtarget_LG[subtarget_LG['rv_date']>=datetime.strptime('2017-02',"%Y-%m")] 
batch2_HW = subtarget_Hw[subtarget_Hw['rv_date']>=datetime.strptime('2017-02',"%Y-%m")]
'''
batch3_LG = subtarget_LG[subtarget_LG['rv_date']>=datetime.strptime('2017-08',"%Y-%m")]
batch3_HW = subtarget_Hw[subtarget_Hw['rv_date']>=datetime.strptime('2017-08',"%Y-%m")]
'''
batch1 = pd.concat([batch1_HW, batch1_LG])
subs2 = pd.concat([batch2_HW, batch2_LG])
batch2 = subs2[subs2['rv_date']<datetime.strptime('2017-08',"%Y-%m")]
batch3 = subs2[subs2['rv_date']>=datetime.strptime('2017-08',"%Y-%m")]

'''
len(batch2[batch2['rv_date']>datetime.strptime('2017-04',"%Y-%m")])
len(batch2[batch2['rv_date']<datetime.strptime('2017-08',"%Y-%m")])
len(batch3)
'''
dill.load_session('start_point.pkl')
dill.dump_session('start_point.pkl')
dill.load_session('batch1_save.pkl')

bash
cd jamr
python3

import dill
dill.load_session('start_point.pkl')
dill.load_session('till_250.pkl')
len(dt3)

subs = batch1
sp = sentence_splitter.SentenceSplitter('en')

bash = pexpect.spawn("/bin/bash")
bash.sendline(". scripts/config.sh")
bash.expect(". scripts/config.sh")

dt1 = pd.DataFrame()
iteration = 0
time_record = datetime.now()
for j in subs.index[146:]:
    ## parse the review and save them in text file.
    st = sp.split(subs.review[j].strip())
    if len(st) == 0 : 
        iteration += 1
        print('\t {0} finished out of {1} at {4}, time took from start {2}, started at {3}'.format(\
            iteration, len(subs),datetime.now()-time_record,time_record,datetime.now()), end = "\r")
        dt3 = dt3.append(pd.DataFrame({"IDX" : subs.X[j],"AMR": [None], 'triple':[None]}))
    else:
        with open('current.txt','w') as cf:
            for k in range(len(st)):
                if len(st[k]) > 1 :
                    _=cf.write(st[k]+"\r\n")
        _ = bash.sendline("\r\n")
        ######### run the parser
        #print("start parser")  
        if os.path.exists('current.txt.parsed'):
            os.remove('current.txt.parsed')
        if os.path.exists('current.txt.parsed.err'):
            os.remove('current.txt.parsed.err')
        _=bash.sendline("scripts/PARSE_IT.sh current.txt")
        time.sleep(1)
        d = 0
        while (os.path.exists('current.txt.tmp') \
            or os.path.exists('current.txt.deps') \
            or not os.path.exists('current.txt.parsed')) :
            d +=1
            time.sleep(1)
        while os.stat('current.txt.parsed').st_size == 0:
            time.sleep(1)
        ## read the parsed results and load to new column? new data frame? . 
        d = 0
        with open("current.txt.parsed","r") as ed:
            test = [l.rstrip() for line in ed for l in line.split("\r\n")]
        with open("current.txt.parsed.err","r") as ed:
            testest = [l.rstrip() for line in ed for l in line.split("\r\n")]
        dt1 = dt1.append(pd.DataFrame({"IDX" : subs.X[j],"AMR": [test], 'triple':[testest]}))
        iteration += 1
        print('\t {0} finished out of {1} at {4}, time took from start {2}, started at {3}'.format(\
            iteration, len(subs),datetime.now()-time_record,time_record,datetime.now()), end = "\r")
        time.sleep(1)
        #os.remove('current.txt.parsed')
        #os.remove('current.txt.parsed.err')
        if iteration % 12 == 0 :        
            if os.path.exists('current.txt.tmp'):
                os.remove('current.txt.tmp')
            if os.path.exists('current.txt.tok'):
                os.remove('current.txt.tok')
            del bash
            bash = pexpect.spawn("/bin/bash")
            _=bash.sendline(". scripts/config.sh")
            _=bash.expect(". scripts/config.sh")


del bash
dill.dump_session('batch1_IT.pkl')
dill.dump_session('batch3_IT.pkl')


dt2 = dt1
dt2.iloc[27,2]

dt2.drop(27)
dt2.reindex = [ i for i in range(74)]


with open("current.txt.parsed","r") as ed:
    test = [l.rstrip() for line in ed for l in line.split("\r\n")]

with open("current.txt.parsed.err","r") as ed:
    testest = [l.rstrip() for line in ed for l in line.split("\r\n")]

dt1 = dt1.append(pd.DataFrame({"IDX" : subs.X[j],"AMR": [test], 'triple':[testest]}))
dt1.iloc[25]
dt1.iloc[:,0]
subs.X.iloc[:15]

# 147 존나김 
del bash
dill.dump_session('batch1_save.pkl')

desc
dt2 = pd.DataFrame()
year = 2016
revs = subtarget_LG
sp = sentence_splitter.SentenceSplitter('en')
time_record = datetime.now()
for i in range(12):
    upper = i+2
    lower = i+1
    year_a = year
    if upper == 13:
        year_a += 1
        upper = 1
    upper_b = revs['rv_date']<datetime.strptime(str(year_a)+"-"+str(upper),"%Y-%m")
    lower_b = revs['rv_date']>=datetime.strptime(str(year)+"-"+str(lower),"%Y-%m")
    print("working on ",str(year)+"-"+str(lower),"number of review : ",len(revs[upper_b&lower_b]))
    subs = revs[upper_b&lower_b]# get the review of selected month.
    for j in subs.index:
        print(subs.review[j])
        ## parse the review and save them in text file. 
        #st = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s|[A-Z].*)',subs.review[j].strip())
        st = sp.split(subs.review[j].strip())
        cf = open('current.txt',"w")
        for k in range(len(st)):
            if len(st[k]) > 1 :
                print('_____write the {0} th line of a'.format(k),str(subs.Product_name[j]), end = "\r")
                cf.write(st[k]+"\r\n")
        cf.close()
        ######### run the parser
        print("start parser")
        bash = pexpect.spawn("/bin/bash")
        bash.sendline(". scripts/config.sh")
        bash.expect(". scripts/config.sh")
        bash.sendline("scripts/PARSE.sh <'current.txt'> output_file> outputsx.txt")
        bash.expect("Decoded", timeout = 600000)
        bash.sendline("echo finishe")
        bash.expect("finishe", timeout= 6000)
        ## read the parsed results and load to new column? new data frame? . 
        time.sleep(3)
        with open("outputsx.txt","r") as ed:
            test = [l for line in ed for l in line.split("\r\n")]
        
        dt2 = dt2.append(pd.DataFrame({"AMR": [test], "IDX" : subs.X[j]}))
    if i == 0 :
        time_records = [{str(year)+"-"+str(lower):datetime.now()-time_record}]
    else :
        time_records.append({str(year)+"-"+str(lower):datetime.now()-time_record})

len(dt2)


len(revs_Hw.Product_name.unique())
len(revs_Sa.Product_name.unique())
len(revs_LG.Product_name.unique())
revs_Hw.columns
##


with open("outputsx.txt","r") as ed:
    test = [l for line in ed for l in line.split("\r\n")]
test
cf = open('current.txt',"w")
for k in range(len(st)):
        print('____write the {0} th line'.format(k), end = "\r")
        #cf.write(st[k]+"\r\n")
cf.write("tts")
cf.close()
        ######### run the parser

st
subs.columns

st = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s|[A-Z].*)',subs.review[j].strip())
cf = open('current.txt',"w")
for k in range(len(st)):
    cf.write(st[k]+"\r\n")
            
cf.close()

ddd = "Pro :First of all, I am a very heavy user. Despite this, the Mate 9 easily has 20-30% battery remaining at 23:00pm the end of each day. I love it, I never have to worry about my phone going dead, or turning down the screen brightness (and compromising usability) just to help the battery life!I have 2 phone numbers , so I had the iPhone 6s and Huawei P9 (one sim slot) before, now I can can use them both in the same phone with the Dual Sim Mate 9 !I love the EMUI, you can split the screen for two APPs running at the same time! I can watch my video from Youtube while chatting with my frineds throught Whatsapp or Wechat. With APP twin, you can login 2 FB accounts at same time!I need to say: the Cameras are awesome too! Two rear cameras : 20M monochrome+ 12M color.  It also has Pro shooting mode- if you can not take a nice photo with the Mate 9's cameras, or if you didn't have the opportunity to use DSLR functions before, now is the time to learn! I will attache some photo samples to this review later.CONs:The phone is big, not very suitable for one-handed use. But maybe this is the size you are looking for !"
st = sp.split(ddd.strip())
cf = open('current.txt',"w")
for k in range(len(st)):
    if len(st[k]) > 1 :
        print('_____write the {0} th line of a'.format(k),str(subs.Product_name[j]), end = "\r")
        cf.write(st[k]+"\r\n")

cf.close()

### Run the performances
bash = pexpect.spawn("/bin/bash")
bash.expect(" ")
bash.before
bash.sendline(". scripts/config.sh")
bash.before
bash.expect("\r\n")
bash.sendline("scripts/PARSE.sh <'current.txt'> output_file> outputsx.txt")
bash.expect("\r\n")
bash.expect("Decoded", timeout = 60000)
bash.before
bash.sendline("echo i'm done")
bash.expect("i'm done")

### get the result from parsed AMR tranfer them into networkx graph. 
with open("outputsx.txt","r") as ed:
    test = [l for line in ed for l in line.split("\r\n")]

test = []
### save the results in pandas. 




