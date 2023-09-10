#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing Libraries
import numpy as np
import pandas as pd
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import calendar
from scipy.stats import skew,kurtosis

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Importing Datasets
df_item = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_order_items_dataset.csv")
df_reviews = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_order_reviews_dataset.csv")
df_orders = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_orders_dataset.csv")
df_products = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_products_dataset.csv")
df_geolocation = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_geolocation_dataset.csv")
df_sellers = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_sellers_dataset.csv")
df_order_pay = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_order_payments_dataset.csv")
df_customers = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/olist_customers_dataset.csv")
df_category = pd.read_csv("/Users/nathanaela.p.k/Downloads/archive/product_category_name_translation.csv")


# In[6]:


#Identification null values in the dataset
def count_null_values(df, dataset_name):
    
    num_of_total_null_values = sum(df.isnull().sum().values)
    print(f"{dataset_name} dataset has {num_of_total_null_values} null values")
    return num_of_total_null_values

customer_null = count_null_values(df_customers, "Olist Customer")
geolocation_null = count_null_values(df_geolocation, "Olist Geolocation")
order_items_null = count_null_values(df_item, "Olist Order Items")
order_payments_null = count_null_values(df_order_pay, "Olist Order Payments")
order_reviews_null = count_null_values(df_reviews, "Olist Order Reviews")
orders_null = count_null_values(df_orders, "Olist Orders")
products_null = count_null_values(df_products, "Olist Products")
sellers_null = count_null_values(df_sellers, "Olist Products")


# In[7]:


#Colomn with null values
def detect_null_columns(df, dataset_name):
    
    col = []
    s = df.isnull().sum()
    for x in range(len(s)):
        if s[x] > 0:
            col.append(s.index[x])
    tot_cols = len(col)
    if tot_cols == 0:
        print(f"{dataset_name} dataset has no null columns")
    else:
        print(f"{dataset_name} dataset has {tot_cols} null columns and they are:")
        for x in col:
            print(x, end=',')
        print()
    return col, len(col)

total_customer_null_cols, customer_null_cols = detect_null_columns(df_customers, "Olist Customer")
total_geolocation_null_cols, geolocation_null_cols = detect_null_columns(df_geolocation, "Olist Geolocation")
total_order_items_null_cols, order_items_null_cols = detect_null_columns(df_item, "Olist Order Items")
total_order_payments_null_cols, order_payments_null_cols = detect_null_columns(df_order_pay, "Olist Order Payments")
total_order_reviews_null_cols, order_reviews_null_cols = detect_null_columns(df_reviews, "Olist Order Reviews")
total_orders_null_cols, orders_null_cols = detect_null_columns(df_orders, "Olist Orders")
total_products_null_cols, products_null_cols = detect_null_columns(df_products, "Olist Products")
total_sellers_null_cols, sellers_null_cols = detect_null_columns(df_sellers, "Olist Products")


# In[10]:


df_orders.info()


# In[35]:


#identification missing values colomn orders
orders = df_orders.dropna()

(df_orders.isna().sum()/len(df_orders)*100).sort_values(ascending=False)


# In[12]:


#Convert the datatype to datetime
df_orders['orders_delivered_carrier_date'] = pd.to_datetime(df_orders.order_delivered_carrier_date)
df_orders['order_estimated_delivery_date'] = pd.to_datetime(df_orders.order_estimated_delivery_date)
df_orders['order_delivered_customer_date'] = pd.to_datetime(df_orders.order_delivered_customer_date)
df_orders['order_delivered_carrier_date'] = pd.to_datetime(df_orders.order_delivered_carrier_date)
df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders.order_delivered_carrier_date)

df_orders.dtypes


# In[37]:


#identification duplicate colomn order
orders[orders.duplicated(keep=False)]


# In[38]:


#identification colomn outlier
orders.describe(include='all')


# In[39]:


#identification incosistent colomn orders
orders['order_status'].unique()


# In[40]:


#final variable name colomn orders
orders_fix = orders


# In[41]:


df_item.info()


# In[42]:


#identification missing values colomn orders
(df_item.isna().sum()/len(df_item)*100).sort_values(ascending=True)


# In[43]:


#identification duplicate colomn order
df_item[df_item.duplicated(keep=False)]


# In[44]:


#Outlier based on stat desc colomn items
df_item.describe(include='all')


# In[17]:


#Outlier use plot
fig, ax = plt.subplots(ncols=2, nrows=1,figsize = (15,6))

sns.histplot(data=df_item, x='price', ax= ax[0])
plt.ylabel('jumlah')

sns.histplot(data=df_item, x='freight_value', ax= ax[1])
plt.ylabel('jumlah')


# In[26]:


#Handling Outlier Colomn Price

#Count Q1 and Q3 colomn price
Q1_price = df_item.price.quantile(0.25)
Q3_price = df_item.price.quantile(0.75)

#Count upper and lower limit colomn price
limit_lower_price = Q1_price - (Q3_price - Q1_price)*1.5
limit_upper_price = Q3_price + (Q3_price - Q1_price)*1.5

item_handling_out_price = df_item

#Count median colomn price
median_price = df_item['price'].median()

#Change outlier with median
item_handling_out_price.loc[item_handling_out_price["price"] > limit_upper_price, "price"]= median_price

#Handling Outlier Colomn freight_value
#Count Q1 and Q3 colomn freight_value
Q1_freight_value = df_item.freight_value.quantile(0.25)
Q3_freight_value = df_item.freight_value.quantile(0.75)

#Count upper and lower limit colomn freight_value
limit_lower_freight_value = Q1_freight_value - (Q3_freight_value - Q1_freight_value)*1.5
limit_upper_freight_value = Q3_freight_value + (Q3_freight_value - Q1_freight_value)*1.5

#Count median colomn freight_value
median_freight_value = df_item['freight_value'].median()

item_handling_out_price_freight = item_handling_out_price

#Change outlier with median
item_handling_out_price_freight.loc[item_handling_out_price_freight["freight_value"] > limit_upper_freight_value, "freight_value"]= median_freight_value


# In[45]:


#Plot distribution after handling outlier
fig, ax = plt.subplots(ncols=2, nrows=1,figsize = (12,8))

sns.histplot(data=item_handling_out_price_freight, x='price', ax= ax[0])

sns.histplot(data=item_handling_out_price_freight, x='freight_value', ax= ax[1])


# In[46]:


#identification incosistent colomn items
item_handling_out_price_freight['order_item_id'].unique()


# In[47]:


#final variable name colomn items
items_fix = item_handling_out_price_freight


# In[48]:


df_order_pay.info()


# In[49]:


df_order_pay.describe(include='all')


# In[50]:


#identification missing values colomn payment
(df_order_pay.isna().sum()/len(df_order_pay)*100).sort_values(ascending=True)


# In[51]:


#identification duplicate colomn payment
df_order_pay[df_order_pay.duplicated(keep=False)]


# In[23]:


#Outlier use plot payment
fig, ax = plt.subplots(figsize = (10,6))

sns.histplot(data=df_order_pay, x='payment_value')


# In[24]:


#Handling Outlier Colomn Payment

#Count Q1 and Q3 colomn payment
Q1_payment = df_order_pay.payment_value.quantile(0.25)
Q3_payment = df_order_pay.payment_value.quantile(0.75)

#Count upper and lower limit colomn price
limit_lower_payment = Q1_payment - (Q3_payment - Q1_payment)*1.5
limit_upper_payment = Q3_payment + (Q3_payment - Q1_payment)*1.5

#Count median colomn price
median_payment = df_order_pay['payment_value'].median()

#Change outlier with median
pay_handling_out = df_order_pay
pay_handling_out.loc[pay_handling_out["payment_value"] > limit_upper_payment, "payment_value"]= median_payment


# In[25]:


#Plot distribution after handling outlier
fig, ax = plt.subplots(figsize = (10,6))

sns.histplot(data=pay_handling_out, x='payment_value')


# In[53]:


#identification incosistent colomn payment_type
pay_handling_out.payment_type.unique()


# In[54]:


#identification incosistent colomn payment_sequential
df_order_pay.payment_sequential.unique()


# In[55]:


#identification incosistent colomn payment_installments
pay_handling_out.payment_installments.unique()


# In[56]:


#final variable name colomn payments
payment_fix = pay_handling_out


# In[57]:


df_products.info()


# In[58]:


df_products.describe(include='all')


# In[29]:


#identifikasi missing value colomn product
(df_products.isna().sum()/len(df_products)*100).sort_values(ascending=False)


# In[30]:


#tabel product have missing value
product_nan = df_products[df_products.isnull().any(axis=1)]
product_nan.head()


# In[34]:


#drop missing value
product_drop_nan = df_products.dropna()

#identification missing value after handling
(product_drop_nan.isna().sum()/len(df_products)*100).sort_values(ascending=False)


# In[59]:


#identification duplicate colomn product
product_drop_nan[product_drop_nan.duplicated(keep=False)]


# In[60]:


#identification outlier colomn product
product_drop_nan.describe(include='all')


# In[61]:


#identification incosistent colomn product
product_drop_nan.head()


# In[63]:


#final variable name colomn product
products_fix = product_drop_nan


# In[64]:


df_category.info()


# In[65]:


df_category.describe(include='all')


# In[66]:


#identification missing values colomn product category
(df_category.isna().sum()/len(df_category)*100).sort_values(ascending=False)


# In[67]:


#identification duplicate colomn product category
df_category[df_category.duplicated(keep=False)]


# In[68]:


#identification incosistent colomn product category
df_category['product_category_name_english'].unique()


# In[69]:


#identification incosistent colomn product category
df_category['product_category_name'].unique()


# In[70]:


#final variable name colomn product
category_products_fix = df_category


# In[71]:


df_customers.info()


# In[72]:


#identification missing values colomn customer
(df_customers.isna().sum()/len(df_customers)*100).sort_values(ascending=False)


# In[73]:


#identification duplicate colomn customer
df_customers[df_customers.duplicated(keep=False)]


# In[74]:


#identification outlier colomn customer
df_customers.describe(include='all')


# In[75]:


#identification incosistent colomn customer
df_customers['customer_state'].unique()


# In[76]:


#final variable name colomn customer
customers_fix = df_customers


# In[77]:


#Objective 1
#merge colomn orders, payments & items
merge_order_item = pd.merge(orders_fix,items_fix, on= 'order_id')
merge_order_items_pay = pd.merge(merge_order_item, payment_fix, on= 'order_id')

#check colomn
merge_order_items_pay = merge_order_items_pay[['order_id','order_status','order_purchase_timestamp','price','freight_value','payment_type','payment_value']]
merge_order_items_pay.head(5)


# In[79]:


#add colomn month
merge_order_items_pay['month'] = merge_order_items_pay['order_purchase_timestamp'].dt.to_period('M')

#grouping
total_penjualan = merge_order_items_pay.groupby('month')['payment_value'].sum()
df_total_penjualan = total_penjualan.reset_index()
df_total_penjualan.columns = ['Bulan', 'Total_Penjualan']
df_total_penjualan['Bulan'] = df_total_penjualan['Bulan'].dt.to_timestamp()
df_total_penjualan


# In[80]:


# visualisasi

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_total_penjualan, x='Bulan', y='Total_Penjualan')
plt.title('Tren Total Penjualan', fontsize= 15)
plt.xlabel('Bulan')
plt.ylabel('Total Penjualan')
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45)
plt.show()


# In[86]:


#Objective 2
def bar_plot(x, y, df, title, xlabel, ylabel, width, height, order = None, rotation=False, palette='winter', hue=None):
    ncount = len(df)
    plt.figure(figsize=(width,height))
    ax = sns.barplot(x = x, y=y, palette=palette, order = order, hue=hue)
    plt.title(title, fontsize=20)
    if rotation:
        plt.xticks(rotation = 'vertical')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    ax.yaxis.set_label_position('left')
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text

    plt.show()


# In[87]:


top_20_cities = customers_fix['customer_city'].value_counts().head(20)
x = top_20_cities.index
y = top_20_cities.values
bar_plot(x,y, customers_fix, 'Top 20 Brazilian cities with most orders', 'City', 'Count', 12,8, rotation=True, palette='flare')


# In[89]:


#Objective 3
method_count = payment_fix['payment_type'].value_counts().to_frame().reset_index()

# Plotly piechart
colors = None
trace1 = go.Pie(labels=method_count['index'], values=method_count['payment_type'],
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
layout = dict(title= "Customer Payment Type", 
              height=400, width=800,)
fig = dict(data=[trace1], layout=layout)
iplot(fig)


# In[90]:


#Objective 4
#merge colomn orders, order_items, products dan product category name translation
merge_order_items_products = pd.merge(merge_order_item, products_fix, on= 'product_id')
merge_order_items_products_cat = pd.merge(merge_order_items_products, category_products_fix, on= 'product_category_name', suffixes=('_order_items', '_product_category' ))

#show colomn
merge_order_items_products_cat = merge_order_items_products_cat[['order_id','price','product_category_name_english']]
merge_order_items_products_cat


# In[91]:


# grouping most seller based on product category
grouping_penjualan_cat = merge_order_items_products_cat.groupby('product_category_name_english').agg({'price':'sum'}).reset_index()
most_seller_product_cat = grouping_penjualan_cat.nlargest(10,'price')
most_seller_product_cat


# In[92]:


#visualization

plt.figure(figsize=(12, 6))
sns.barplot(data=most_seller_product_cat, x='product_category_name_english', y='price')
plt.title('Top 10 Product Categories by Total Product Sold', fontsize= 15)
plt.xlabel('Kategori produk')
plt.ylabel('Total Penjualan')
plt.show()


# In[ ]:




