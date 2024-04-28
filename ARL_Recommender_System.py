
#### Association Rule Based Recommender System
######################################################

## İş Problemi
###################
# Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine en
# uygun ürün önerisini birliktelik kuralı kullanarak yapınız. Ürün önerileri 1 tane
# ya da 1'den fazla olabilir. Karar kurallarını 2010-2011 Germany müşterileri
# üzerinden türetiniz.
# Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

## Veri Seti Hikayesi
###########################
# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış
# işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi
# mevcuttur.

# 8 Değişken - 541.909 Gözlem - 45.6MB
# InvoiceNo -- Fatura Numarası (Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder)
# StockCode -- Ürün kodu (Her bir ürün için eşsiz)
# Description -- Ürün ismi
# Quantity -- Ürün adedi (Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate -- Fatura tarihi
# UnitPrice -- Fatura fiyatı (Sterlin)
# CustomerID -- Eşsiz müşteri numarası
# Country -- Ülke ismi

## Proje Görevleri
########################

### Görev 1: Veriyi Hazırlama
###################################

import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",500)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr", False) # # çıktını tek satırda olmasını sağlar.
from mlxtend.frequent_patterns import apriori, association_rules

# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_excel("4.) Recommendation Systems/Projects (including Bonus)/Bonus/online_retail_II-220908-163413/online_retail_II.xlsx", sheet_name= "Year 2010-2011")
df = df_.copy()

df.head()
df.shape
df.info()
df.describe().T
## Quantity ve Price değişkenlerinde - değerler mevcut ve max değerleri çok uçuk.

#*** Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
df[df["StockCode"]=="POST"].index
len(df[df["StockCode"]=="POST"].index)
df.drop(df[df["StockCode"]=="POST"].index, axis=0,inplace=True) ## ***
len(df[df["StockCode"]=="POST"].index)
df.shape

#*** Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df.isnull().sum()
df.dropna(inplace=True) ## ***
df.shape

#*** Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
df = df[~df["Invoice"].str.contains("C", na=False)] ## ***
df.shape

# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df = df[df["Price"]>0]
df.shape

# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
df.describe().T
sbn.boxplot(df[["Quantity"]])
plt.title("Quantity")
plt.show(block=True)
sbn.boxplot(df["Price"])
plt.title("Price")
plt.show(block=True)

## Baskılayalım:
def outlier_thresholds(dataframe, variable):
    quantile1 = dataframe[variable].quantile(0.01)
    quantile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quantile3 - quantile1
    low_limit = quantile1 - interquantile_range*1.5
    up_limit = quantile3 + interquantile_range*1.5
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")

## Aşağıdaki fonk. kullanılabilirdi!
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Price"] > 0]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df.describe().T ## max değerleri düzeltilmiş ve negatif değerlerden arındırılmış.
sbn.boxplot(df[["Quantity"]])
plt.title("Quantity")
plt.show(block=True)
sbn.boxplot(df["Price"])
plt.title("Price")
plt.show(block=True)

df.head()

### Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
#########################################################################

## Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.
# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON   SPACEBOY BABY GIFT SET…
# Invoice
# 536370                  0                             1                         0
# 536852                  1                             0                         1
# 536974                  0                             0                         0
# 537065                  1                             0                         0
# 537463                  0                             0                         1

df.Country.value_counts()
df_ger = df[df["Country"]=="Germany"]
df_ger.shape
###!!!!Hatalı!!! df_ger.pivot_table(index=["Invoice"], columns=["Description"], values="Quantity").head()
df_ger.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)

## Aşağıdaki fonk. kullanalım:
def create_invoice_product_df(dataframe, id=False):
    if id:
        # faturadaki ürünün satın alınma miktarı
        return dataframe.groupby(["Invoice","StockCode"])["Quantity"].sum().\
            unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)
    else:
        # faturadaki ürünün satın alınma miktarı
        return dataframe.groupby(["Invoice","Description"])["Quantity"].sum().\
            unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)

ger_inv_pro_df = create_invoice_product_df(df)
ger_inv_pro_df.iloc[95:100, 29:49]

## Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"]==stock_code][["Description"]].values[0].tolist()
    print(product_name)

## Kural oluşturacağımız Fonksiyonumuz:
def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe["Country"]==country]
    dataframe = create_invoice_product_df(dataframe, id)
    # Ürünlerin frekansını kullanarak olasılıklarını(Supportunu) hesaplama:
    frequent_itemsets = apriori(dataframe.astype("bool"), min_support=0.01,
                                use_colnames=True, low_memory=True)
    # Olası ürün çiftlerinin frekansını kullanarak olasılıklarını hesaplama:
    rules = association_rules(frequent_itemsets, metric="support",min_threshold=0.01)
    return rules

df = df.copy()
# Ülke seçimi
create_rules(df, country= "Germany")
# Veri hazırlığı
df = retail_data_prep(df)
# Birliktelik kurallarının oluşturulması
rules = create_rules(df)
rules.describe().T
# filtreleyerek çağıralım
rules[(rules["support"]>0.05) & (rules["confidence"]>0.5) & (rules["lift"]>5)].\
    sort_values("confidence",ascending=False)


### Görev 2: Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
# Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.
check_id(df, 21987)
check_id(df, 23235)
check_id(df, 22747)

sorted_rules = rules.sort_values("lift",ascending=False)

# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift",ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j==product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

arl_recommender(rules, 21987, 2)
arl_recommender(rules, 23235, 4)
arl_recommender(rules, 22747, 2)

# Adım 3: Önerilecek ürünlerin isimlerine bakınız.

check_id(df_ger, 22745)
check_id(df_ger, 22748)
check_id(df_ger, 21988)
check_id(df, 23238)

