import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
# import openpyxl
# import plotly.express as px
# import plotly.graph_objects as go
import requests
from urllib.parse import quote,unquote
import re
import csv
# from lxml import etree
import os
import time
import pandas as pd
from PIL import Image
import io
# import cv2
from urllib.parse import quote,unquote
# from sentence_transformers import SentenceTransformer, util
# from sklearn.manifold import TSNE
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.metrics.pairwise import cosine_similarity
# import hypertools as hyp
import random
# from paddleocr import PaddleOCR, draw_ocr# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换   # 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
# ocr = PaddleOCR(use_angle_cls=True, lang="ch") # need to run only once to download and load model into memory
# from keybert import KeyBERT
# kw_model = KeyBERT()



###################################### main part ########################################################

st.set_page_config(page_title='Product Information Inquiry', page_icon=':bar_chart:', layout='wide')

st.write('-----------------------------------')
col1, col2 = st.columns([2,2])
with col1:
    title = st.radio('请选择月份', ['val_jan_23','val_feb_23','val_mar_23','val_apr_23','val_may_23','val_jun_23','val_jul_23','val_aug_23','val_sep_23','val_oct_23','val_nov_23','val_dec_23'],horizontal=True)
with col2:
    upload_models= st.file_uploader('请上传你需要的产品 xlsx格式',type=['csv'])    
st.write('-----------------------------------')


# model2=pd.read_csv(upload_models)
model=pd.read_csv(upload_models)
st.write(model)
model=model.fillna(0)

model.loc[:,'vol_jan_22':'vol_dec_22']=model.loc[:,'vol_jan_22':'vol_dec_22'].astype('int')


# model['test'] = model['vol_jan_22'] + model['vol_feb_22']

model['price']= model[title]/model[title.replace('val','vol')].astype('int')

model['price']=model['price'].apply(lambda x : '1)<3000' if x<=3000 else
                                                         ('2)3000 < 5000' if x>3000 and x<=5000 else
                                                         ('3)5000 < 7000' if x>5000 and x<=7000 else
                                                         ('4)7000 < 10000' if x>7000 and x<=10000 else
                                                         ('5)10000 < 15000' if x>10000 and x<=15000 else
                                                         ('6)>15000' if x>15000 else ''))))))
mid=model['price']   #取备注列的值
model.pop('price')  #删除备注列
model.insert(5,'price',mid) #插入备注列
model=model.fillna(0)


models=model
# st.write(models['val_jan_23'].values.tolist(), models['val_dec_23'].values.tolist())
# models['test'] = models['val_jan_23'] + models['val_dec_23']

for i in models.loc[:,'vol_jan_22':'vol_dec_22'].columns.values.tolist():
    models['YTD_'+i]=models.loc[:,'vol_jan_22':i].sum(axis=1)

for i in models.loc[:,'val_jan_22':'val_dec_22'].columns.values.tolist():
    models['YTD_'+i]=models.loc[:,'val_jan_22':i].sum(axis=1)

for i in models.loc[:,'vol_jan_23':'vol_dec_23'].columns.values.tolist():
    models['YTD_'+i]=models.loc[:,'vol_jan_23':i].sum(axis=1)

for i in models.loc[:,'val_jan_23':'val_dec_23'].columns.values.tolist():
    models['YTD_'+i]=models.loc[:,'val_jan_23':i].sum(axis=1)
    

facts=['vol','val','vol_share','val_share','+/- vol_share','+/- val_share']
months_vol=['vol_jan_22','vol_feb_22','vol_mar_22','vol_apr_22','vol_may_22','vol_jun_22','vol_jul_22','vol_aug_22','vol_sep_22','vol_oct_22','vol_nov_22','vol_dec_22',
            'vol_jan_23','vol_feb_23','vol_mar_23','vol_apr_23','vol_may_23','vol_jun_23','vol_jul_23','vol_aug_23','vol_sep_23','vol_oct_23','vol_nov_23','vol_dec_23',]
months_val=['val_jan_22','val_feb_22','val_mar_22','val_apr_22','val_may_22','val_jun_22','val_jul_22','val_aug_22','val_sep_22','val_oct_22','val_nov_22','val_dec_22',
            'val_jan_23','val_feb_23','val_mar_23','val_apr_23','val_may_23','val_jun_23','val_jul_23','val_aug_23','val_sep_23','val_oct_23','val_nov_23','val_dec_23',]

ytds_vol=['YTD_vol_jan_22','YTD_vol_feb_22','YTD_vol_mar_22','YTD_vol_apr_22','YTD_vol_may_22','YTD_vol_jun_22','YTD_vol_jul_22','YTD_vol_aug_22','YTD_vol_sep_22','YTD_vol_oct_22','YTD_vol_nov_22','YTD_vol_dec_22',
          'YTD_vol_jan_23','YTD_vol_feb_23','YTD_vol_mar_23','YTD_vol_apr_23','YTD_vol_may_23','YTD_vol_jun_23','YTD_vol_jul_23','YTD_vol_aug_23','YTD_vol_sep_23','YTD_vol_oct_23','YTD_vol_nov_23','YTD_vol_dec_23',]
ytds_val=['YTD_val_jan_22','YTD_val_feb_22','YTD_val_mar_22','YTD_val_apr_22','YTD_val_may_22','YTD_val_jun_22','YTD_val_jul_22','YTD_val_aug_22','YTD_val_sep_22','YTD_val_oct_22','YTD_val_nov_22','YTD_val_dec_22',
          'YTD_val_jan_23','YTD_val_feb_23','YTD_val_mar_23','YTD_val_apr_23','YTD_val_may_23','YTD_val_jun_23','YTD_val_jul_23','YTD_val_aug_23','YTD_val_sep_23','YTD_val_oct_23','YTD_val_nov_23','YTD_val_dec_23',]

month_combine_vol=['vol_jan_23','vol_feb_23','vol_mar_23','vol_apr_23','vol_may_23','vol_jun_23','vol_jul_23','vol_aug_23','vol_sep_23','vol_oct_23','vol_nov_23','vol_dec_23',]
month_combine_val=['val_jan_23','val_feb_23','val_mar_23','val_apr_23','val_may_23','val_jun_23','val_jul_23','val_aug_23','val_sep_23','val_oct_23','val_nov_23','val_dec_23',]

markets=['offline channel','Model','price','region','city_tier']
products=['Brand']

col1, col2, col3, col4, col5, col6 = st.columns(6, gap='large')
with col1:
    months_vol=st.multiselect('请选择一个您要查询单月的 Volume:',
                    (months_vol),) #(single_months)
with col2:
    months_val=st.multiselect('请选择一个您要查询单月的 Value:',
                    (months_val), ) #(ytds)
with col3:
    ytds_vol=st.multiselect('请选择一个您要查询的 YTD Volume:',
                    (ytds_vol),) #(single_months)
with col4:
    ytds_val=st.multiselect('请选择一个您要查询的 YTD Value:',
                    (ytds_val), ) #(ytds)
with col5:
    month_combine_vol=st.multiselect('请选择一个您要查询的 Combined月的 Vol',
                    (month_combine_vol),) #(single_months)
    newname_vol = st.text_input('newname_vol ', 'e.g. vol_feb & mar_23')
with col6:
    month_combine_val=st.multiselect('请选择一个您要查询的 Combined月的 Value',
                    (month_combine_val), ) #(ytds)
    newname_val = st.text_input('newname_val ', 'e.g. val_feb & mar_23')

st.write('-----------------------------------')

col1, col2, col3 = st.columns(3)
with col1:
    fact=st.multiselect('请选择一个你要分析的维度:',
                    (facts), 
                    (facts))
with col2:
    market=st.multiselect('请选择一个你要分析的市场维度:',
                    (markets), 
                    (markets[0:3]))
with col3:
    product=st.multiselect('请选择一个您要查询的产品:',
                    (products), 
                    (products))

models['sum_' + newname_vol] =models[month_combine_vol].sum(axis=1)
models['sum_' + newname_val] =models[month_combine_val].sum(axis=1)

st.write('检查一下合并后的数据')
st.dataframe(models, use_container_width=True)

st.write('-----------------------------------')
#用 ‘vol_jan_22'作为起始点，把数据选出来
period_values=models.loc[:,'vol_jan_22':].columns.values.tolist() 
# st.write(months_vol)

table_sales=pd.pivot_table(models, index=product,columns=market, values=period_values, aggfunc=np.sum,margins=True)
table_share=table_sales.apply(lambda x : x/x.sum()*100,axis=0)

#把multi-index 的 header 改掉
table_sales.columns = [i[0]+'_'+i[1] for i in table_sales.columns.values.tolist()]
table_share.columns = [i[0]+'_'+i[1] for i in table_share.columns.values.tolist()]

# st.write(table_sales,table_share)

st.write('Sales value 的 full table')
st.dataframe(table_sales, use_container_width=True)
st.write('Value share 的 full table')
st.dataframe(table_share, use_container_width=True)

st.write('-----------------------------------')
#根据选择日期，把sales value和share 提取出来
period_total=months_vol+months_val+ytds_vol+ytds_val+month_combine_vol+month_combine_val

col1, col2, col3 = st.columns(3,gap='large')
with col1:
    st.write('period_total',period_total)
with col2:
    st.write('fact',fact)
with col3:
    st.write('market',models[market[0]].unique().tolist()+['All'])

#把选取的日期和market 做一个表头，用作提取同样日期的数据
period_market=[]  

for m in models[market[0]].unique().tolist()+['All']:
    # st.write('m:', m)
    period_market = period_market + [p + '_' + m for p in period_total]

col1, col2 = st.columns(2,gap='large')
with col1:
    st.write('period x market:', period_market)
with col2:
    st.write('period x market:', [p for p in period_market if p.count('23')]) # 把做shr change 的columns 取出来
    st.write('fact',fact)


for f in fact:
    
    if f=='vol' or f=='val':
        table_sales_output=table_sales[period_market] # 把所有日期 的 sales value 做出来

    elif f=='vol_share' or f=='val_share':
        table_share_output=table_share[period_market] # 把所有日期 的 value share 做出来

    elif f=='+/- vol_share' or f=='+/- val_share':
        # st.write('把做shr change 的columns',[p for p in period_market if p.count('23')])
        period_share_chg = [p for p in period_market if p.count('23')] # 把做shr change 的columns 取出来
        period_new_forshrchg = [p+'_ +/- shr' for p in period_share_chg] # 创建新的columns

        for p in period_share_chg:
            table_share[p+'_ +/- shr'] = table_share[p] - table_share[p.replace('23','22')] 

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.write(p)
        # with col2:
        #     st.write(p.replace('23','22'))

        table_share_chg_output=table_share[period_new_forshrchg]

st.subheader('Crosstable _ Sales Value')
st.dataframe(table_sales_output)
st.subheader('Crosstable _ Value Share')
st.dataframe(table_share_output)
st.subheader('Crosstable _ +/- Value Share')
st.dataframe(table_share_chg_output.round(2)


