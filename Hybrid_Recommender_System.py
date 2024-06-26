#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr",False)

# Adım 1: Movie ve Rating veri setlerini okutunuz.

# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv("4.) Recommendation Systems/Projects (including Bonus)/HybridRecommender-221114-235254/datasets/movie.csv")

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv("4.) Recommendation Systems/Projects (including Bonus)/HybridRecommender-221114-235254/datasets/rating.csv")

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

df = movie.merge(rating, how="left", on="movieId")
df.head()

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız. Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullandığını hesaplıyoruz.
df["title"].value_counts()

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz.
type(df["title"].value_counts()) # pandas.core.series.Series
## Dataframe olarak kaydetmeliyiz.
rating_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = rating_counts[rating_counts["title"]<1000].index

df = df[~df["title"].isin(rare_movies)]
df["title"].value_counts()

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
df.columns
user_movie_df = df.pivot_table(index="userId", columns="title", values="rating")
user_movie_df.head()

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("4.) Recommendation Systems/recommender_systems-221002-142715/recommender_systems/datasets/movie_lens_dataset/movie.csv")
    rating = pd.read_csv("4.) Recommendation Systems/recommender_systems-221002-142715/recommender_systems/datasets/movie_lens_dataset/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 5000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns= ["title"], values="rating")
    return user_movie_df


#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
df["userId"].value_counts()
userId = "34576.0"
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index==random_user]

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
### dataframe'i sadece izlenen filmlerle sınırlı kıl.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df

## Nan

#**** Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.
# *** satırın tüm değerlerini toplayabilmek için:
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

user_movie_count[user_movie_count["movie_count"]>=30].sort_values("movie_count",ascending=False)
percentage = len(movies_watched) * 60/100 # 19.8
users_same_movies = list(user_movie_count[user_movie_count["movie_count"]>=percentage].index)
len(users_same_movies)

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile
# benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde
# movies_watched_df dataframe’ini filtreleyiniz.
movies_watched_df.head()
type(movies_watched_df.index.isin(users_same_movies)) ## true-false numpy.ndarray
type([movies_watched_df.index.isin(users_same_movies)]) ## true-false list
## random user'sız filtre dataframe:
movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
## random user'a ait dataframe:
random_user_df[movies_watched]
##*** random userlı dataframe:
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                     random_user_df[movies_watched]])

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
type(corr_df) ## pandas.core.series.Series
## corr_df'i DataFrame'e çevirelim:
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

# corr_df[corr_df["user_id_1"] == random_user]
# corr_df[corr_df["user_id_2"] == random_user]

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan)
# kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["corr"]>0.65) & (corr_df["user_id_1"]==random_user)][["user_id_2", "corr"]].reset_index(drop=True)
top_users.sort_values("corr", ascending=False)

# Adım 4:  top_users dataframe’ini rating veri seti ile merge ediniz.
## columns isim değişikliği:
top_users.rename(columns={"user_id_2":"userId"}, inplace=True)
## merge:
top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
top_users_rating[top_users_rating["userId"]==random_user] ## boş küme

#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating":"mean"})
recommendation_df = recommendation_df.reset_index()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
recommendation_df[recommendation_df["weighted_rating"]>3.5].sort_values("weighted_rating", ascending=False)
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"]>3.5].sort_values("weighted_rating", ascending=False)

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])


#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width",None)
pd.set_option("display.expand_frame_repr", False)

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv("4.) Recommendation Systems/Projects (including Bonus)/HybridRecommender-221114-235254/datasets/movie.csv")
rating = pd.read_csv("4.) Recommendation Systems/Projects (including Bonus)/HybridRecommender-221114-235254/datasets/rating.csv")

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
df = movie.merge(rating, how="left", on="movieId")
user_latest_5rated_movieId = int(df.loc[(df["userId"]==random_user) & (df["rating"]==5.0)].
                                 sort_values(by="timestamp", ascending=False)
                                 [:1]["movieId"].values)

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie_name_df = user_movie_df[user_movie_df.index==user_latest_5rated_movieId]


# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_name_df).sort_values(ascending=False).head(10)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.

def item_based_recommender(movie_Id, user_movie_df):
    movie_name = user_movie_df[df[df["movieId"]==movie_Id]["title"].values[0]]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(11).iloc[1:11]

item_based_recommender(user_latest_5rated_movieId, user_movie_df)


