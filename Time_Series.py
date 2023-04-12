

# Smoothing Methods (Holt-Winters)

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings('ignore')


# Zaman Serisi Problemi: Atmosferdeki Karbondioksitin "Mart 1958 - Aralık 2001" yılları arasındaki gözlemleri mevcut olup gözlem sonundaki 1 ay sonraki periyotta hava kirliliğinin tahminini yapalım:

data = sm.datasets.co2.load_pandas()
y = data.data # hedef değişken

# Hafatalık formattan aylık formata çevirelim:
y = y['co2'].resample('MS').mean()


# Eksik değerler kendisinden önceki/sonraki değerler ile ya da ortalamaları ile doldurulabilir:
y.isnull().sum()

# bir sonraki değeri kullanarak eksik değerleri dolduralım:
y = y.fillna(y.bfill())


y.plot(figsize=(15, 6))
plt.show()
# figure_1: seride trend vardır, mevsimsellik var gibi, durağan değildir.



# veri setini train test olarak ayıralım ve modelin test setindeki başarısına bakalım:
train = y[:'1997-12-01']
len(train)  # 478 ay

# 1998'ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':]
len(test)  # 48 ay



# Zaman Serisi Yapısal Analizi
# Durağanlık Testi (Dickey-Fuller Testi)

def is_stationary(y):
    # "HO: Non-stationary"
    # "H1: Stationary"
    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(y)
# p-value, 0.05 den büyük H0 reddedilemez, yani durağan değildir.


# Zaman Serisi Bileşenleri ve Durağanlık Testi
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)
# figure_2: toplamsal modelin bileşenleri; Level bileşeni, trend, mevsimsellik ve artıkların gösterimini görmekteyiz.
# Serinin level ortalama değerlerini gözlemleriz, yukarı yönlü trendi var, mevsimselliği var. Artıkların ortalaması 0' ın etrafında toplanmış.
# Serinin, durağan olmadığını görüyoruz.



# ********************************************************************************************************************
# Single Exponential Smoothing (SES)
# SES = Level
# Trend ve mevsimselliği yakalayamaz.

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

y_pred = ses_model.forecast(48)

# Elde ettiğimiz tahminleri gerçek değerler ile kıyaslamak için;
mean_absolute_error(test, y_pred)
# ortalama mutlak hatamız 5.70

train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
# çok kötü bir tahminde bulunmuşuz

# 1985 sonrasına bakalım:
train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()



def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

plot_co2(train, test, y_pred, "Single Exponential Smoothing")

ses_model.params
# {'smoothing_level': 0.5,
#  'smoothing_trend': nan,
#  'smoothing_seasonal': nan,
#  'damping_trend': nan,
#  'initial_level': 316.4419309772571,
#  'initial_trend': nan,
#  'initial_seasons': array([], dtype=float64),
#  'use_boxcox': False,
#  'lamda': None,
#  'remove_bias': False}



# smooting_level parametresinde hiperparametre optimizasyonu yapalım:
# Hyperparameter Optimization

def ses_optimizer(train, alphas, step=48):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)

# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka

ses_optimizer(train, alphas)
# farklı alpha değerine karşılık mae değerleri:
best_alpha, best_mae = ses_optimizer(train, alphas)



# Final SES Model
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(48)

plot_co2(train, test, y_pred, "Single Exponential Smoothing")
# Bir tık yükselmiş olsa da hala çok kötü, modelin tahmin performansı: 4.54



# ********************************************************************************************************************
# Double Exponential Smoothing (DES)
# DES: Level (SES) + Trend

# Zaman serileri toplamsal ya da çarpımsal olabilir:
# y(t) = Level + Trend + Seasonality + Noise (Toplamsal model) mevsimsellik ve artık bileşenleri trend den bağımsızsa seri toplamsaldır.
# y(t) = Level * Trend * Seasonality * Noise (çarpımsal model) mevsimsellik ve artık bileşenleri trend e göre şekilleniyorsa seri çarpımsaldır.
# Mevsimsellik ve Artıklar(hatalar)ın ortalaması 0 ın etrafında rastgele dağılıyorsa toplamsaldır.

ts_decompose(y)
# figure :

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
# alpha değeri, level bileşeninde geçmiş gerçek değerlere mi, geçmiş tahmini değerlere mi ağırlık vereceğimiz.
                                                         smoothing_trend=0.5)
# beta değeri, yakın trende mi, uzak trende mi ağırlık vereceğimiz.

y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")
# trend olsa da mevsimsellik olmadığı için DES iyi vbir başarı sergileyemedi.

# Hyperparameter Optimization
############################
def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)
# best_alpha: 0.01 best_beta: 0.71 best_mae: 1.7411

# Final DES Model
############################
final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                               smoothing_slope=best_beta)
# trend:"mul" yapılarak çarpımsal modele de bakılabilir.
y_pred = final_des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")
# mae:1.74 ancak mevsimselliği yakalayamıyoruz



# **********************************************************************************************************************
# Triple Exponential Smoothing (Holt-Winters)
# TES = SES + DES + Mevsimsellik
# Level, trend ve mevsimsellik etkilerini değerlendirerek tahmin yapmaktadır.

tes_model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5,
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)

y_pred = tes_model.forecast(48)
plot_co2(train, test, y_pred, "Triple Exponential Smoothing")



# Hyperparameter Optimization

alphas = betas = gammas = np.arange(0.20, 1, 0.10)

abg = list(itertools.product(alphas, betas, gammas))


def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)
# Çıktı: best_alpha: 0.8 best_beta: 0.5 best_gamma: 0.7 best_mae: 0.6177



# Final TES Model

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")
# MAE: 0.62



