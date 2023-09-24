import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


st.title('白酒数据分析')
st.markdown("""
使用了Python与scrapy爬虫框架研发了数据收集软件，重点采集了微博上的白酒相关数据。根据百度指数的需求图谱、百度搜索相关词进行筛选，找到了如下的关键词：
- 白酒
- 五粮液
- 茅台酒
- 泸州老窖
- 洋河
- 汾酒
- 郎酒
- 习酒
- 金六福
- 西凤酒
- 剑南春
- 蒸馏酒
- 酱香
- 烧酒
- 老白干
""")

with st.sidebar:
    st.write("关注公众号：行恒编程1对1")
    st.image("static/xingheng.jpg")
    st.markdown("查找更多答案，官网： <https://pychegg.51zhi.com/>")


wine_df = pd.read_json("wine.json", lines=True)
wine_df['year'] = wine_df['created_at'].apply(lambda x: x.year)
wine_df = wine_df.astype({'year': 'string'})

st.metric("总数据条目", len(wine_df))
    
year_df = wine_df.groupby('year', as_index=False).count()
year_df['n_blog'] = year_df['ad_marked']
year_df = year_df[['year', 'n_blog']]
print("year blog", year_df.dtypes)

col1, col2 = st.columns(2)
with col1:
    # group by year
    st.subheader('网络讨论热度年度变化趋势')
    st.line_chart(year_df, x='year', y='n_blog')
    #st.table(year_df)

wine_product_df = pd.read_csv('wine_product.csv')
wine_product_df = wine_product_df.astype({'year': 'string'})
print("wine product", wine_product_df.dtypes)

with col2:
    st.subheader('白酒年度产量数据')
    st.line_chart(wine_product_df, x='year', y='product')


wine_product_and_blog_df = wine_product_df.merge(year_df, on="year", how='left')
wine_product_and_blog_df = wine_product_and_blog_df.fillna(0)
print(wine_product_and_blog_df)
st.table(wine_product_and_blog_df)

st.subheader('线性回归模型检测相关性')
st.text("从以下数据可以看出模型的拟和度低于50%，并且n_blog的p-value大于0.164，所以两者之间的相关性不好")
cov_model = smf.ols('product ~ n_blog', data=wine_product_and_blog_df).fit()
st.code(cov_model.summary())


st.subheader('ARIMA产量预测模型')
st.write("基于ARIMA模型来预测未来的白酒产量，使用了python库pmdarima的auto_arima来训练模型，然后进行预测。最终取得最好的模型参数是SARIMAX(0, 1, 1)，并且成功预测了未来一年的白酒产量是582. ")
#model = ARIMA(wine_product_and_blog_df['product'], order=(1, 0, 0))
model = pm.auto_arima(wine_product_and_blog_df['product'], start_p=1, start_q=1, stepwise=True, d=1)
st.code(model.summary())

prediction, confint = model.predict(n_periods=1, return_conf_int=True)
st.write("2024产量置信区间预测")
st.table(confint)
st.write('2024产量预测')
st.table(prediction)