import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from poetry_dep_regr.data import df
import scipy.stats as stats
import numpy as np
from tabulate import tabulate
from math import sqrt as sqrt
from statistics import mean
import matplotlib.pyplot as plt

class basest:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = LinearRegression(fit_intercept=True, copy_X=True).fit(self.x.to_numpy().reshape(-1, 1), self.y)
        self.y_pred = self.model.predict(self.x.to_numpy().reshape(-1, 1))
        self.unbiasedness = mean(self.y - self.y_pred)
        self.alpha = 0.05
        self.n = len(self.y)
        self.m = 1
        self.df_fact = self.n - 1
        self.df_ost = self.n - self.m - 1
        self.y_mean = np.mean(self.y)
        self.SS_fact = np.sum((self.y_pred-self.y_mean)**2)
        self.SS_ost = np.sum((self.y-self.y_pred)**2)
        self.MS_fact = self.SS_fact / self.m
        self.MS_ost = self.SS_ost/self.df_ost
        self.MS_total = np.sum((self.y-self.y_mean)**2)/self.df_fact
        self.F_critical = stats.f.ppf(1 - self.alpha, dfn=self.m, dfd=self.df_ost)
        self.T_critical = stats.t.ppf(1 - self.alpha / 2, self.df_ost)
        self.SS_x = np.sum((self.x - np.mean(self.x))**2)
        self.SEb0 = sqrt(self.MS_ost * (1/self.n + (np.mean(self.x))**2 / self.SS_x))
        self.SEb1 = sqrt(self.MS_ost / self.SS_x)
        self.lingres_ = None
        self.spear_kendl_ = None
        self.statistical_reliability_F_ = None
        self.comp_sample_averages_ = None
        self.var_ratio_ = None
        self.average_approx_err_ = None
        self.scheme_of_variance_analysis_ = None
        self.T_ratio_regr_ = None
        self.T_ratio_corr_ = None
        self.confidence_intervals_ = None
        self.y_pred_plus_5_ = None
        self.individual_confidence_intervals_ = None
        self.elasticity_coefficient_ = None
        self.turning_points_test_ = None
        self.check_residuals_sum_ = None
        self.spearman_test_ = None
        self.goldfeld_quandt_test_ = None
        self.durbin_watson_test_ = None
        self.rs_criterion_ = None
        self.nonlinear_models_ = None

    @property
    def lingres(self):
        if self.lingres_ is None:
            lingres_result = stats.linregress(self.x, self.y)
            self.lingres_ = { # коэф в урав лин регр + коэф детерминации + коэф пирсона + ...
            "slope" : lingres_result.slope,
            "intercept" : lingres_result.intercept,
            "rvalue": lingres_result.rvalue,
            "R^2": lingres_result.rvalue**2,
            "pvalue": lingres_result.pvalue,
            "stderr": lingres_result.stderr,
            "intercept_stderr" : lingres_result.intercept_stderr,
        }
        return self.lingres_ 
    
    @property
    def spear_kendl(self):
        if self.spear_kendl_ is None:
           self.spear_kendl_ = { # коэф корр
            "spearman" : self.x.corr(self.y, method='spearman'), 
            "kendall" : self.x.corr(self.y, method='kendall')
        }    
        return self.spear_kendl_ 
    
    @property
    def statistical_reliability_F(self):
        if self.statistical_reliability_F_ is None:
            MSR = self.SS_fact/self.m
            MSE = self.SS_ost/self.df_ost
            F = MSR/MSE
            #F = self.lingres["R^2"]/(1-self.lingres["R^2"]) * (self.df_ost)
            self.statistical_reliability_F_ = {
                "F-fact": F,
                "F-critical": self.F_critical,
            }
        return self.statistical_reliability_F_

    @property
    def average_approx_err(self): # средняя ошибка аппроксимации
        if self.average_approx_err_ is None: 
            self.average_approx_err_ = (1/self.n) * np.sum(np.abs((self.y - self.y_pred)/self.y)) * 100
        return self.average_approx_err_
    
    @property    
    def scheme_of_variance_analysis(self):
        if self.scheme_of_variance_analysis_ is None: 
            F = self.MS_fact/self.MS_ost
            p_value = 1 - stats.f.cdf(F, self.m, self.df_ost)
            data = [["factorial", self.m, self.SS_fact, self.MS_fact, F, self.F_critical, F > self.F_critical, f"p_value: {p_value}"],
            ["residual", self.df_ost, self.SS_ost, self.MS_ost, "", "", "", ""],
            ["total", self.m+self.df_ost, self.SS_fact+self.SS_ost, self.MS_total, "", "", "", ""]]
            self.scheme_of_variance_analysis_ = tabulate(pd.DataFrame(data, columns=["Sources of variation", "DF", "SS", "MS", "F - criteria", "F - critical", "F - criteria > F - critical", "p_value"]), headers="keys", tablefmt="grid")
        return self.scheme_of_variance_analysis_
    
    @property 
    def T_ratio_regr(self):
        if self.T_ratio_regr_ is None:

            # t-статистики
            t0 = self.lingres["intercept"] / self.SEb0
            t1 = self.lingres["slope"] / self.SEb1

            # Критическое значение t-критерия
            tcrit = self.T_critical

            # p-value для коэффициентов
            p_value0 = 2 * (1 - stats.t.cdf(abs(t0), self.df_ost))
            p_value1 = 2 * (1 - stats.t.cdf(abs(t1), self.df_ost))

            # Проверка значимости коэффициентов
            tcomp0 = abs(t0) > tcrit
            tcomp1 = abs(t1) > tcrit

            self.ratio_regr_ = {
                "T_intercept": t0,
                "T_slope": t1,
                "T_critical": tcrit,
                "p_intercept": p_value0,
                "p_slope": p_value1,
                "T_intercept_Tcr_copmp": tcomp0,
                "T_slope_Tcr_comp": tcomp1,
            }
        return self.ratio_regr_
    
    @property 
    def T_ratio_corr(self):
        if self.T_ratio_corr_ is None:
            # t-статистика
            t = (self.lingres["rvalue"] * sqrt(self.df_ost)) / sqrt(1 - self.lingres["rvalue"]**2)

            # Критическое значение t-критерия

            tcrit = self.T_critical

            # p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t), self.df_ost))

            # Проверка значимости
            tcomp = abs(t) > tcrit

            self.T_ratio_corr_ = {
                "T_corr": t,
                "T_crit": tcrit,
                "p": p_value,
                "T_corr_Tcr_comp": tcomp
            }

        return self.T_ratio_corr_
    
    @property
    def confidence_intervals(self):
        if self.confidence_intervals_ is None:
            tcrit = self.T_critical
            ci_b0 = [self.lingres["intercept"]-tcrit*self.SEb0, self.lingres["intercept"]+tcrit*self.SEb0]
            ci_b1 = [self.lingres["slope"]-tcrit*self.SEb1, self.lingres["slope"]+tcrit*self.SEb1]
            self.confidence_intervals_ = {
                "confidence_intervals_b0": ci_b0,
                "confidence_intervals_b1": ci_b1
            }
        return self.confidence_intervals_
    
    @property
    def y_pred_plus_5(self):
        if self.y_pred_plus_5_ is None:
            x_mean = np.mean(self.x)
            x_new = x_mean * 1.05  # Увеличение на 5%
            y_pred_new = self.model.predict(np.array([[x_new]]))[0]

            # Стандартная ошибка прогноза
            se_pred = sqrt(self.MS_ost * (1 + 1/self.n + (x_new - x_mean)**2 / self.SS_x))

            # Доверительный интервал для прогноза
            t_crit = self.T_critical
            ci_lower = y_pred_new - t_crit * se_pred
            ci_upper = y_pred_new + t_crit * se_pred

            self.y_pred_plus_5_ = {
            "x_new": x_new,
            "y_pred_new": y_pred_new,
            "confidence_interval": [float(ci_lower), float(ci_upper)]
            }
        return self.y_pred_plus_5_

    @property
    def individual_confidence_intervals(self):
        if self.individual_confidence_intervals_ is None:
            x_array = self.x.to_numpy()
            se_ind = sqrt(self.MS_ost * (1 + 1/self.n + np.sum((x_array - np.mean(x_array))**2) / self.SS_x))
            t_crit = self.comp_sample_averages["T-critical"]
            ci_lower = self.y_pred - t_crit * se_ind
            ci_upper = self.y_pred + t_crit * se_ind
            self.individual_confidence_intervals_ = list(zip(ci_lower, ci_upper))
        return self.individual_confidence_intervals_

    @property
    def elasticity_coefficient(self):
        if self.elasticity_coefficient_ is None:
            x_mean = np.mean(self.x)
            y_mean = np.mean(self.y)
            slope = self.lingres["slope"]
            self.elasticity_coefficient_ = slope * (x_mean / y_mean)
        return self.elasticity_coefficient_

    @property
    def turning_points_test(self):
        if self.turning_points_test_ is None:
            """Критерий поворотных точек для проверки случайности остатков."""
            residuals = self.y - self.y_pred
            n = len(residuals)
            turning_points = 0
            for i in range(1, n - 1):
                if (residuals[i] > residuals[i - 1] and residuals[i] > residuals[i + 1]) or \
                (residuals[i] < residuals[i - 1] and residuals[i] < residuals[i + 1]):
                    turning_points += 1

            expected_turning_points = 2 * (n - 2) / 3
            variance_turning_points = (16 * n - 29) / 90
            z = (turning_points - expected_turning_points) / sqrt(variance_turning_points)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            self.turning_points_test_ = {
                "turning_points": turning_points,
                "z_score": z,
                "p_value": p_value
            }
        return self.turning_points_test_

    @property
    def check_residuals_sum(self):
        if self.check_residuals_sum_ is None:
            """Проверка, что сумма остатков равна нулю."""
            residuals = self.y - self.y_pred
            self.check_residuals_sum_ = np.sum(residuals)
        return self.check_residuals_sum_

    @property
    def spearman_test(self):
        if self.spearman_test_ is None:
            """Тест Спирмена для проверки гомоскедастичности."""
            residuals = self.y - self.y_pred
            rank_x = stats.rankdata(self.x)
            rank_residuals = stats.rankdata(np.abs(residuals))
            corr, p_value = stats.spearmanr(rank_x, rank_residuals)
            self.spearman_test_ = {
                "spearman_correlation": corr,
                "p_value": p_value
            }
        return self.spearman_test_

    @property
    def goldfeld_quandt_test(self):
        if self.goldfeld_quandt_test_ is None:
            """Тест Голдфелда-Квандта для проверки гомоскедастичности."""
            sorted_indices = np.argsort(self.x)
            n = len(self.x)
            split = n // 3
            y1 = self.y[sorted_indices[:split]]
            y2 = self.y[sorted_indices[-split:]]
            x1 = self.x[sorted_indices[:split]].to_numpy().reshape(-1, 1)
            x2 = self.x[sorted_indices[-split:]].to_numpy().reshape(-1, 1)

            model1 = LinearRegression().fit(x1, y1)
            model2 = LinearRegression().fit(x2, y2)
            residuals1 = y1 - model1.predict(x1)
            residuals2 = y2 - model2.predict(x2)

            ssr1 = np.sum(residuals1**2)
            ssr2 = np.sum(residuals2**2)
            f_stat = ssr2 / ssr1
            p_value = 1 - stats.f.cdf(f_stat, split - 2, split - 2)

            self.goldfeld_quandt_test_ = {
                "f_statistic": f_stat,
                "p_value": p_value
            }
        return self.goldfeld_quandt_test_

    @property
    def durbin_watson_test(self):
        if self.durbin_watson_test_ is None:
            """Критерий Дарбина-Уотсона для проверки автокорреляции остатков."""
            residuals = self.y - self.y_pred
            self.durbin_watson_test_ = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        return self.durbin_watson_test_

    @property
    def rs_criterion(self):
        if self.rs_criterion_ is None:
            """RS-критерий для проверки нормальности остатков."""
            residuals = self.y - self.y_pred
            r = np.max(residuals) - np.min(residuals)
            s = np.std(residuals, ddof=1)
            self.rs_criterion_ = r / s
        return self.rs_criterion_

    @property
    def nonlinear_models(self):
        if self.nonlinear_models_ is None:
            """Построение нелинейных регрессионных моделей."""
            x = self.x.to_numpy().reshape(-1, 1)
            y = self.y

            # Полиномиальная регрессия (степень 2)
            poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            poly_model.fit(x, y)
            y_pred_poly = poly_model.predict(x)

            # Сравнение моделей
            mse_linear = np.mean((self.y_pred - y)**2)
            mse_poly = np.mean((y_pred_poly - y)**2)

            self.nonlinear_models_ = {
                "mse_linear": mse_linear,
                "mse_poly": mse_poly,
                "best_model": "Полиномиальная" if mse_poly < mse_linear else "Линейная"
            }
        return self.nonlinear_models_

    def print_summary(self):
        """Выводит все подсчитанные данные в удобном формате."""
        print("=" * 50)
        print("Линейная регрессия:")
        print(tabulate(pd.DataFrame([self.lingres]), headers="keys", tablefmt="grid"))
        
        print("\nКоэффициенты корреляции:")
        print(tabulate(pd.DataFrame([self.spear_kendl]), headers="keys", tablefmt="grid"))

        print("\nСтатистическая надежность результатов регрессионного моделирования:")
        print(tabulate(pd.DataFrame([self.statistical_reliability_F]), headers="keys", tablefmt="grid"))

        print("\nВариация:")
        data = {
        "Показатель": ["SS_x (Сумма квадратов x)", "SS_ост (Остаточная сумма квадратов)", "SS_факт (Факторная сумма квадратов)"],
        "Значение": [self.SS_x, self.SS_ost, self.SS_fact]
        }
        print(tabulate(pd.DataFrame(data), headers="keys", tablefmt="grid"))

        print("\nСредняя ошибка аппроксимации:")
        print(f"{self.average_approx_err:.2f}%")

        print("\nСхема дисперсионного анализа:")
        print(self.scheme_of_variance_analysis)

        print("\nПроверка значимости коэффициентов регрессии:")
        print(tabulate(pd.DataFrame([self.T_ratio_regr]), headers="keys", tablefmt="grid"))

        print("\nПроверка значимости коэффициента корреляции:")
        print(tabulate(pd.DataFrame([self.T_ratio_corr]), headers="keys", tablefmt="grid"))

        print("\nПрогноз при увеличении фактора на 5%:")
        print(tabulate(pd.DataFrame([self.y_pred_plus_5]), headers="keys", tablefmt="grid"))

        print("\nДоверительные интервалы для индивидуальных значений:")
        print(tabulate(pd.DataFrame([self.confidence_intervals["confidence_intervals_b1"]], 
                            columns=["Нижняя граница", "Верхняя граница"]), 
               tablefmt="grid"))

        print("\nСредний коэффициент эластичности:")
        print(f"{self.elasticity_coefficient:.4f}")

        print("\nПроверка случайности остатков (критерий поворотных точек):")
        print(tabulate(pd.DataFrame([self.turning_points_test]), headers="keys", tablefmt="grid"))

        print("\nСумма остатков:")
        print(f"{self.check_residuals_sum:.4f}")

        print("\nТест Спирмена для гомоскедастичности:")
        print(tabulate(pd.DataFrame([self.spearman_test]), headers="keys", tablefmt="grid"))

        print("\nТест Голдфелда-Квандта для гомоскедастичности:")
        print(tabulate(pd.DataFrame([self.goldfeld_quandt_test]), headers="keys", tablefmt="grid"))

        print("\nКритерий Дарбина-Уотсона для автокорреляции:")
        print(f"{self.durbin_watson_test:.4f}")

        print("\nRS-критерий для нормальности остатков:")
        print(f"{self.rs_criterion:.4f}")

        print("\nСравнение линейной и полиномиальной моделей:")
        print(tabulate(pd.DataFrame([self.nonlinear_models]), headers="keys", tablefmt="grid"))

        print("=" * 50)