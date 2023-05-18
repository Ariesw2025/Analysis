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


model=pd.read_csv(upload_models)
model=model.fillna(0)

st.write(model.columns.values.tolist())
st.write(model["vol_jan_22"])

model['price']= model[title]/model[title.replace('val','vol')]
model['price']=model['price'].apply(lambda x : '1)<3000' if x<=3000 else
                                                         ('2)3000 < 5000' if x>3000 and x<=5000 else
                                                         ('3)5000 < 7000' if x>5000 and x<=7000 else
                                                         ('4)7000 < 10000' if x>7000 and x<=10000 else
                                                         ('5)10000 < 15000' if x>10000 and x<=15000 else
                                                         ('6)>= 15000' if x>15000 else ''))))))


mid=model['price']   #取备注列的值
model.pop('price')  #删除备注列
model.insert(5,'price',mid) #插入备注列
model=model.fillna(0)

models=model
models.loc[:,'vol_jan_22':'vol_dec_22']=models.loc[:,'vol_jan_22':'vol_dec_22'].astype('float')

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
st.dataframe(table_sales, use_container_width=True)
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
st.dataframe(table_share_chg_output)






# for i in period_total:
#     st.write(i)
#     i + '_' + 


# months_22=[j for j in single_months if j.count('22')]
# months_23=[j for j in single_months if j.count('23')]

# single_month_value = []
# for s in single_month:
#     single_month_value.append('vol_'+s)
#     single_month_value.append('val_'+s)

# st.write(single_month_value)

# for y in ytd:
#     if y.count('22'):
#         vol_columns=['vol_'+cols for cols in months_22[: months_22.index(ytd[0])+1]]
#         models['vol_ytd_'+y]=models[vol_columns].sum(axis=1)

#         val_columns=['val_'+cols for cols in months_22[: months_22.index(ytd[0])+1]]
#         models['val_ytd_'+y]=models[val_columns].sum(axis=1)

#         single_month_value.append('vol_ytd_'+y)
#         single_month_value.append('val_ytd_'+y)

#     elif y.count('23'):
#         vol_columns=['vol_'+cols for cols in months_23[: months_23.index(ytd[0])+1]]
#         models['vol_ytd_'+y]=models[vol_columns].sum(axis=1)

#         val_columns=['val_'+cols for cols in months_23[: months_23.index(ytd[0])+1]]
#         models['val_ytd_'+y]=models[val_columns].sum(axis=1)

#         single_month_value.append('vol_ytd_'+y)
#         single_month_value.append('val_ytd_'+y)

# st.write(models)

# st.write('market: ',market,single_month_value)

# for mkt in market:
#     table=pd.pivot_table(models, index=product[0],values=single_month_value, aggfunc=np.sum, margins=True)


# # table.columns=[i[1]+'_'+i[0] for i in table.columns.values.tolist()]
# st.write(table)






# for mkt in market:
#     st.write(mkt)
#     table=pd.pivot_table(models, index=product[0],columns=[mkt],values=single_month)


# fact_value = ['vol' if x.count('vol') else 'val' if x.count('val') else '' for x in fact]
# st.write([x for x in set(fact_value)]) #set()函数用于将列表转换为集合，去重后的列表中只有一个元素，即原列表中不同的元素 ['vol', 'val', 'vol', 'val', 'vol']








# models['price']= models['price_feb_23'].apply(lambda x : '1)<3000' if x<=3000 else
#                                                          ('2)3000 < 5000' if x>3000 and x<=5000 else
#                                                          ('3)5000 < 7000' if x>5000 and x<=7000 else
#                                                          ('4)7000 < 10000' if x>7000 and x<=10000 else
#                                                          ('5)10000 < 15000' if x>10000 and x<=15000 else
#                                                          ('6)>= 15000' if x>15000 else ''))))))



# ###################################### share in total market after * 100 ########################################################

# title = st.radio('选择分析维度',('offline channel','Type','price'), key='渠道，产品型号，价格待续',horizontal=True)
# # st.write(title)
# header=title

# share_min = 0.3

# @st.cache_data
# def get_ttl_mkt_share(models,header):
    
#     headers=pd.pivot_table(models, index='Brand',columns=header, 
#                             values=['val_mar_22'],aggfunc=np.sum, margins=True) # 把header的顺序提出来
#     col_name =[col[1] for col in headers.columns.tolist()]  # headers.columns = 

#     models['share_ttl_mkt_22']=models['val_mar_22'].apply(lambda x : (x/(models['val_mar_22'].sum())*100))
#     models['share_ttl_mkt_23']=models['val_mar_23'].apply(lambda x : (x/(models['val_mar_23'].sum())*100))
    
#     table_col_sales=pd.pivot_table(models, index='Brand',columns=header, 
#                             values=['share_ttl_mkt_22','share_ttl_mkt_23'],aggfunc=np.sum, margins=True)
    
#     # table2=table_col_sales.apply(lambda x: x/x.sum()*2*100,axis=0) # 计算列百分比
#     # st.write('full_table for data check', table_col_sales)
#     table = table_col_sales
#     table=table.fillna(0)

#     table.columns = [col[0]+'_'+col[1] for col in table.columns.tolist()]   # clean pivot_table 的columns
#     col_len = int(len(table.columns)/2)

#     st.write(col_name)    
#     for i in range(0,col_len):
#         table[col_name[i]] = table.iloc[:,i+col_len] - table.iloc[:,i]  
#     st.write('full_table for data check', table)
    
#     table2=table.iloc[:-1,:] #把最下面的All 去掉

#     table2=table2[(table2['All']>share_min) | (table2['All']<-share_min)]
#     table2.loc['others']=table[(table['All']<share_min) & (table['All']>-share_min)].sum()
#     table2.sort_values('All',ascending=False,inplace=True)

#     table2_long = table2
#     table2_short = table2.iloc[:,-col_len:]

#     table_col_sales.columns = [col[0]+'_'+col[1] for col in table_col_sales.columns.tolist()]  # clean pivot_table 的columns

#     return table2_long, table2_short, table_col_sales

# table2_ttl_mkt_long, table2_ttl_mkt_short, sales_value =get_ttl_mkt_share(models,header)

# # st.subheader('share_ttl_mkt_long after * 100 ')
# # st.dataframe(table2_ttl_mkt_long.round(5),use_container_width=True) #'share_ttl_mkt_ after * 100 ',
# st.subheader('share_ttl_mkt_short after * 100 ')
# st.dataframe(table2_ttl_mkt_short.round(5),use_container_width=True)
# # st.dataframe('sales _ttl_mkt_ in mio RMB', sales_value/1000000)


# df=table2_ttl_mkt_short.round(1)
# fig = go.Figure()

# for i in df.index:
#     # st.write(i)
#     # st.write(df[df.index==i].values.tolist())
#     fig.add_trace(go.Bar(x=df.columns.values.tolist(),
#                         y=df[df.index==i].values.tolist()[0],name=i,#marker=dict(color=colors[i]), 
#                         text=df[df.index==i].values.tolist()[0], width=[0.5,0.5]))    #,facet_col='vol',, marker_color='color'

# fig.update_layout(barmode='relative',height=600,width=800,plot_bgcolor='ghostwhite')
# fig.update_traces(textposition="inside",textfont_size=15)
# fig.update_xaxes(tickfont_size=12,color='red')


# col1, col2 = st.columns([2,4],gap='large')
# with col1:
#     st.subheader('share_ttl_mkt_short after * 100 ')
#     st.dataframe(table2_ttl_mkt_short.round(1),use_container_width=True,height=550)
# with col2:
#     st.subheader('Chart share_ttl_mkt_short after * 100 ')
#     st.plotly_chart(fig,theme="streamlit", use_container_width=True, height=200)

# ###################################### share within channel after * 100 ########################################################

# # header='offline channel'

# @st.cache_data
# def get_col_share(models,header):
    
#     headers=pd.pivot_table(models, index='Brand',columns=header, 
#                             values=['val_mar_22'],aggfunc=np.sum, margins=True) # 把header的顺序提出来
#     col_name =[col[1] for col in headers.columns.tolist()]  # headers.columns = 

#     # 把pivot table 做出来
#     table_col_sales=pd.pivot_table(models, index='Brand',columns=header, 
#                             values=['val_mar_22','val_mar_23'],aggfunc=np.sum, margins=True)
    
#     # 计算列百分比
#     table=table_col_sales.apply(lambda x: x/x.sum()*2*100,axis=0) # 计算列百分比
#     table=table.fillna(0)

#     # clean pivot_table 的columns
#     table.columns = [col[0]+'_'+col[1] for col in table.columns.tolist()]  # clean pivot_table 的columns 
#     col_len = int(len(table.columns)/2)

#     # clean pivot_table 的columns
#     # print(col_name)    
#     for i in range(0,col_len):
#         table[col_name[i]] = table.iloc[:,i+col_len] - table.iloc[:,i]  
#     st.write('full_table for data check', table.round(3))

#     table2=table.iloc[:-1,:] #把最下面的All 去掉

#     table2=table2[(table2['All']>share_min) | (table2['All']<-share_min)]
#     table2.loc['others']=table[(table['All']<share_min) & (table['All']>-share_min)].sum()
#     table2.sort_values('All',ascending=False,inplace=True)

#     table2_long = table2
#     table2_short = table2.iloc[:,-col_len:]

#     table_col_sales.columns = [col[0]+'_'+col[1] for col in table_col_sales.columns.tolist()]    # clean pivot_table 的columns

#     return table2_long, table2_short, table_col_sales

# table2_long, table2_short, sales_value =get_col_share(models,header)


# # st.subheader('share within channel after * 100 ')
# # st.dataframe(table2_long.round(3),use_container_width=True)
# # st.subheader('share within channel after * 100 ')
# # st.dataframe(table2_short.round(3),use_container_width=True)
# # display('sales within channel in mio RMB', sales_value/1000000)

# df=table2_short.round(1)
# fig = go.Figure()

# for i in df.index:
#     # st.write(i)
#     # st.write(df[df.index==i].values.tolist())
#     fig.add_trace(go.Bar(x=df.columns.values.tolist(),
#                         y=df[df.index==i].values.tolist()[0],name=i,#marker=dict(color=colors[i]), 
#                         text=df[df.index==i].values.tolist()[0], width=[0.5,0.5]))    #,facet_col='vol',, marker_color='color'

# fig.update_layout(barmode='relative',height=600,width=800, plot_bgcolor='ghostwhite')
# fig.update_traces(textposition="inside",textfont_size=15)
# fig.update_xaxes(tickfont_size=12,color='red')
# # fig.update_yaxes(range=[-5,10])


# col1, col2 = st.columns([2,4],gap='large')
# with col1:
#     st.subheader('share_ttl_mkt_short after * 100 ')
#     st.dataframe(table2_short.round(1),use_container_width=True,height=550)
# with col2:    
#     st.subheader('Chart share_ttl_mkt_short after * 100 ')
#     st.plotly_chart(fig,theme="streamlit", use_container_width=True, height=200)

# ###################################### SKU share within channel after * 100 ########################################################
# @st.cache_data
# def get_sku_share(models,header):    
#     table_col_sales=pd.pivot_table(models, index=['Brand','Model'],columns=header, 
#                             values=['val_mar_22','val_mar_23'],aggfunc=np.sum, margins=True)

#     headers=pd.pivot_table(models, index=['Brand','Model'],columns=header, 
#                             values=['val_mar_22'],aggfunc=np.sum, margins=True) # 把header的顺序提出来
#     col_name =[col[1] for col in headers.columns.tolist()]  # headers.columns = 

#     # 把pivot table 做出来
#     table_col_sales=pd.pivot_table(models, index=['Brand','Model'],columns=header, 
#                             values=['val_mar_22','val_mar_23'],aggfunc=np.sum, margins=True)

#     # 计算列百分比
#     table=table_col_sales.apply(lambda x: x/x.sum()*2*100,axis=0) # 计算列百分比
#     table=table.fillna(0)

#     # clean pivot_table 的columns
#     table.columns = [col[0]+'_'+col[1] for col in table.columns.tolist()]  # clean pivot_table 的columns 
#     col_len = int(len(table.columns)/2)

#     # clean pivot_table 的columns
#     # print(col_name)    
#     for i in range(0,col_len):
#         table[col_name[i]] = table.iloc[:,i+col_len] - table.iloc[:,i]  
#     # st.write('full_table for data check', table.round(3))

#     table2_long = table
#     table2_short = table.iloc[:,-col_len:]
#     table_col_sales.columns = [col[0]+'_'+col[1] for col in table_col_sales.columns.tolist()]    # clean pivot_table 的columns

#     return table2_long, table2_short, table_col_sales

# table2_sku_share_long, table2_sku_share_short, sales_value =get_sku_share(models,header)


# # st.subheader('SKU share_long within channel after * 100 ')
# # st.dataframe(table2_sku_share_long.round(3),use_container_width=True)
# # st.subheader('SKU share_short within channel after * 100 ')
# # st.dataframe(table2_sku_share_short.round(3),use_container_width=True)

# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv()#.encode('utf-8')

# csv = convert_df(table2_sku_share_short)

# st.download_button(
#     label="Download data table2_sku_share_short_all sku",
#     data=csv,
#     file_name='table2_sku_share_short',
#     mime='csv',
# )

# ###################################### SKU share within channel after * 100 ########################################################

# brand_list=table2_short[table2_short.index!='others'].index.values.tolist()
# brand = st.radio('请选择你要看的品牌',brand_list,horizontal=True)

# @st.cache_data
# def get_sku_details(table2_sku_share_long,sales_value,models,brand):
#     col_len = int(len(sales_value.columns)/2)
#     table2_short_sku=table2_sku_share_long.iloc[:,(-col_len-1):]
#     table2_short_sku.reset_index(inplace=True)
#     table2_short_sku=table2_short_sku[table2_short_sku['Brand']==brand]

#     sku = models[models['Brand']==brand]
#     table_sku_price=pd.pivot_table(sku, index=['Model'],columns=header, 
#                                 values=['price_feb_23'],aggfunc=np.mean)
#     table_sku_price=table_sku_price.round(0)
#     table_sku_price.reset_index(inplace=True)
#     table_sku_price.columns = [col[0]+col[1] for col in table_sku_price.columns.tolist()]
#     table_sku_price['url']=table_sku_price['Model'].apply(lambda x:'https://search.jd.com/Search?keyword=' + x + '&enc=utf-8&wq=' + x + '&pvid=da294005df284f6eb670940cd0660f69')
#     table2_sku_detail = pd.merge(table2_short_sku,table_sku_price, on='Model')
    
#     return table2_sku_detail

# table2_sku_detail=get_sku_details(table2_sku_share_long,sales_value,models,brand)

# st.subheader('SKU share with price within channel after * 100 ')
# st.dataframe(table2_sku_detail.round(3),use_container_width=True)

# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv()#.encode('utf-8')

# csv = convert_df(table2_sku_detail.round(3))

# st.download_button(
#     label='Download data'+' SKU details_' + brand,
#     data=csv,
#     file_name='table2_sku_share_short_' + brand,
#     mime='csv',)

