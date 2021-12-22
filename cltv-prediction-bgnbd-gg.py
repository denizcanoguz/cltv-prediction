##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################
##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################
# pip install lifetimes
# pip install sqlalchemy
# conda install -c anaconda mysql-connector-python
# conda install -c conda-forge mysql
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#########################
# Verinin Veri Tabanından Okunması
#########################
# database: group_3 user: group_3 password: miuul host: 34.79.73.237 port: 3306
# credentials.
creds = {'user': 'group_3',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_3'}
# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))
# conn.close()
pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)
retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
retail_mysql_df.Country
retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()
df = retail_mysql_df.copy()
#########################
# Veri Ön İşleme
#########################
# Ön İşleme Öncesi
df = df[df["Country"].str.contains("United Kingdom", na=False)] # UK müsterilerinin seçimi
df.Country.unique()
df.describe().T
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)] # faturada c ile başlayanları veri setimizde almadık. # İADELER
df = df[df["Quantity"] > 0] # En az 1 satın alım olanları seçtik
df = df[df["Price"] > 0] # 0 dan fazla kazandıranları seçtik
replace_with_thresholds(df, "Quantity") # burada normalde baskılama yöntemi kullanıyoruz.
replace_with_thresholds(df, "Price")    # burada normalde baskılama yöntemi kullanıyoruz.
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"].max() # en son fatura tarihimiz
today_date = dt.datetime(2011, 12, 11) # analiz tarihimiz
#########################
# Lifetime Veri Yapısının Hazırlanması
#########################
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce analiz gününe göre, burada kullanıcı özelinde)
# T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç
cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days, # recency hasaplama: Son satın alma üzerinden geçen zaman. Haftalık.
                                                         lambda date: (today_date - date.min()).days], # T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
                                         'Invoice': lambda num: num.nunique(), # frequency : tekrar eden toplam satın alma sayısı unique
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})  # monetary : satın alma başına ortalama kazanç
cltv_df.columns = cltv_df.columns.droplevel(0) # hiyerarşik column yapısını düzeltiyoruz
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df["recency"] = cltv_df["recency"] / 7 # haftalık almak için yapılan işlem
cltv_df["T"] = cltv_df["T"] / 7
# cltv_df["frequency"] = cltv_df["frequency"].astype(int) # yukarıdaki son 2 satırda hata alırsak bu kısmı çalıştırın

##############################################################
# 2. BG-NBD Modelinin Kurulması
##############################################################
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])
########################
# "expected_purc_1_week"
########################
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])
#########################
# "expected_purc_1_month"
#########################
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

###########################
# "expected_average_profit"
###########################
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
#########################################################################################################################
# Görev 1:
# 6 aylık CLTV Prediction
# ▪     2010-2011 UK müşterileri için 6 aylık CLTV prediction yapınız.
# ▪ Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız..
#########################################################################################################################
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head()
cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="scaled_clv", ascending=False).head()


##############################################################
# Görev 2:
# Farklı zaman periyotlarından oluşan CLTV analizi
# ▪     2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# ▪ 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
# ▪     Fark var mı? Varsa sizce neden olabilir?
##############################################################
# 1 AYLIK CLTV HESAPLANMASI
cltv_1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_1.head()
cltv_1 = cltv_1.reset_index()
cltv_1.sort_values(by="clv", ascending=False).head(10)
cltv_1_final = cltv_df.merge(cltv_1, on="CustomerID", how="left")
cltv_1_final.sort_values(by="clv", ascending=False).head(10)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_1_final[["clv"]])
cltv_1_final["scaled_clv_1ay"] = scaler.transform(cltv_1_final[["clv"]])
cltv_1_final.sort_values(by="scaled_clv_1ay", ascending=False).head()
bir_aylik = cltv_1_final.sort_values(by="scaled_clv_1ay", ascending=False).head()

# 12 AYLIK CLTV HESAPLANMASI
cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_12.head()
cltv_12 = cltv_12.reset_index()
cltv_12.sort_values(by="clv", ascending=False).head(10)
cltv_12_final = cltv_df.merge(cltv_12, on="CustomerID", how="left")
cltv_12_final.sort_values(by="clv", ascending=False).head(10)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_12_final[["clv"]])
cltv_12_final["scaled_clv_12ay"] = scaler.transform(cltv_12_final[["clv"]])
cltv_12_final.sort_values(by="scaled_clv_12ay", ascending=False).head()
oniki_aylik = cltv_12_final.sort_values(by="scaled_clv_12ay", ascending=False).head()
###################################################################################################################################
# 1 AYLIK VE 12 AYLIK DEĞERLERİN İLK 10 DEĞERİNİN KARŞILAŞTIRILMASI
# >> Fark var gözüküyor. Modelimize göre 1 aylık seçimde 1 ay için tahmin yapılıyor ,12 aylık seçimde 12 ay için tahmin yapılıyor
# bazı müşteriler örüntülerini tamamlamadığı için 1.aya göre alt sıralara düşmüş olabilirler
##################################################################################################################################
bir_aylik.head(10)
oniki_aylik.head(10)



##################################################################################################################################
# Görev 3:
#Segmentasyon ve Aksiyon Önerileri
# ▪     2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba
#       (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# ▪     4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon
# önerilerinde bulununuz
##################################################################################################################################

##########################################
# CLTV'ye Göre Segmentlerin Oluşturulması
##########################################
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()
cltv_final.sort_values(by="scaled_clv", ascending=False).head()
pd.set_option('display.max_columns', None)
cltv_final.groupby("segment").agg({"mean","count", "sum"})

#### <Aksiyon Önerilerim> ####
# >> D segmenti bizim en eski müşterilerimiz barındırıyor. Hatta bizden ortalama olarak en son alışveriş yapan segment. Ancak bize en az kazandıran segment.
# >> Bu segment için önerim kendimizi onlara sık sık hatırlatmak olur. Mail ve sms yoluyla kapanyalarımız ve yeni ürünlerimizden sık sık haberdar edebiliriz.

# >> C segmenti aynı D segmenti gibi bizim eski müşterilerimizden oluşuyor ancak, D segmentine göre bize daha çok kazandırmışlar.
# >> Bu segment için önerim Onlara yine mail ve sms yoluyla ulaşıp en sık aldıkları ürünlerde indirimler sunmamız olabilir.





##############################################################
# Görev 4:
#Veri tabanına kayıt gönderme
# ▪ Aşağıdaki değişkenlerden oluşacak final tablosunu veri tabanına gönderiniz.
# ▪     tablonun adını isim_soyisim şeklinde oluşturunuz.
# ▪      Tablo ismi ilgili fonksiyonda "name" bölümüne girilmelidir.
# CustomerID | recency | T | frequency | monetary |  expected_purc_1_week |
# expected_purc_1_month |  expected_average_profit | clv | scaled_clv | segment
##############################################################
pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
cltv_final.head()
cltv_final["CustomerID"] = cltv_final["CustomerID"].astype(int)

cltv_final.to_sql(name='denizcan_oguz', con=conn, if_exists='replace', index=False)

pd.read_sql_query("select * from denizcan_oguz", conn)














