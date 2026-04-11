# Homework 5 — Expanded presentation content

**Use this for the 3–5 minute presentation and Gradescope write-up.**  
Sections are in order. **Highlight** lines are the most impactful takeaways—stress these when presenting.

---

## 1. Assumptions

We needed a stationary series to fit AR/ARMA/ARMAX, but we did **not** assume stationarity—we **tested** it. The Augmented Dickey–Fuller test on the raw Boston housing series showed it was **non-stationary** (trend). After OLS linear detrending, we re-ran the ADF test on the residuals and treated the detrended series as stationary enough to model. So stationarity was something we **checked**, not assumed.

We also implicitly assumed **linearity**: the relationship between the series and its lags (and between the series and the interest rate in ARX/ARMAX) is linear. The model class is linear in the lags. For the **credit–housing link**, we assumed that housing prices are influenced by credit conditions, so the 30-year mortgage rate should relate to prices or their changes. We didn’t assume that blindly: for Boston 2010–2017 we computed **corr(Δ price, interest rate level) = −0.40 (p ≈ 0)**—a statistically significant moderate negative correlation—and **corr(Δ price, Δ rate) = 0.17 (p = 0.11)**, which is not significant. So the **level** of the rate matters more than its month-to-month change, and we then evaluated whether ARX/ARMAX actually improved out-of-sample MSE over AR.

> **Highlight:** We **tested** stationarity (ADF) and the credit link (correlations and AR vs ARX/ARMAX); we didn’t assume them. The number to remember: **r = −0.40 (p ≈ 0)** for change in price vs interest rate level.

---

## 2. Trend removal

We removed the trend so the residual could be treated as stationary. We used **OLS linear detrending**: regress the series on a time index and use the **residuals** as the detrended series, then validated with the ADF test on those residuals. Other options include **differencing** (e.g. first difference for a unit root), higher-order differencing, or fitting a polynomial or spline trend and subtracting it. We’d choose **linear OLS detrend** again when the trend looks roughly linear—it’s simple, interpretable, and easy to add back for forecasts—or **differencing** when the ADF suggests a unit root and we care about growth rates.

> **Highlight:** We used **OLS linear detrend**, validated with **ADF on residuals**. For a similar project we’d do the same when the trend is roughly linear, or use differencing if the series has a unit root.

---

## 3. Generalization (new dataset, similar prediction task)

For a new dataset and a similar prediction task we’d: (1) **Explore**—plot the series, check for missing values, and define the prediction target (level vs change, horizon). (2) **Stationarity**—run the **Augmented Dickey–Fuller test**; if the series is non-stationary, detrend or difference and re-test. (3) **Correlations and lags**—run basic correlations (e.g. target vs exogenous variable) and, from the course, look at **cross-correlation** between the two signals (or power spectrum / cross-spectrum) to see **which lags are meaningful**, and **test the statistical significance** of those lags, not just their size. (4) **Model class**—start with AR using ACF/PACF, then ARMA/ARMAX if we have an exogenous variable. (5) **Order selection**—use AIC/BIC and/or out-of-sample MSE (e.g. rolling one-step-ahead). (6) **Evaluate**—hold out a test period and report MSE, RMSE, or MAE and compare models (e.g. AR vs ARX). We’d use **AR/ARMA** when we only have one series, and **ARX/ARMAX** when we have a plausible exogenous driver that improves out-of-sample performance. Dataset type and **domain knowledge** both matter—frequency, trend/seasonality, and choice of exogenous variable (e.g. mortgage rate for housing).

> **Highlight:** Same pipeline: **plot → ADF for stationarity → cross-correlation / significance of lags → AR/ARMA or ARX/ARMAX → AIC/BIC and out-of-sample MSE.** Test which lags are **statistically significant**, not just large.

---

## 4. Evaluation metrics

For **prediction accuracy** we focused on **MSE**, **RMSE**, and **MAE**: MSE penalizes large errors more, RMSE is on the same scale as the variable, and MAE is more robust to outliers. For **model choice** we used **AIC** and **BIC** to balance fit and complexity and to select orders *p* and *q*. In our project we used in-sample **MSE** and **AIC/BIC** to choose AR and ARX/ARMAX orders, and **out-of-sample MSE** (with a short-term, one-step-ahead rolling refit) to compare AR vs ARX vs ARMAX.

> **Highlight:** **MSE, RMSE, MAE** for accuracy; **AIC and BIC** for model and order selection. We used both in-sample (AIC/BIC, MSE) and out-of-sample MSE to compare models.

---

## 5. Exogenous inputs (ARX vs AR only)

We compared **AR(p)** with **ARX(p)** and **ARMAX(p,q)** using the 30-year mortgage rate as the exogenous variable, on both in-sample and **out-of-sample MSE** with short-term prediction. ARX gave similar or slightly better test MSE than AR in our runs. ARX can help when the interest rate actually drives part of house price variation (credit channel); it can add little when the relationship is weak or when the rate is already captured by recent price lags. Our takeaway: we’d still use ARX when the interest rate is available and interpretable, given our correlation result and the comparison we did.

> **Highlight:** **ARX with the interest rate** gave similar or slightly better **out-of-sample MSE** than AR. We’d use ARX when the exogenous variable is available and interpretable.


---

## 6. Error sources

Main sources of error: **data** (measurement or definition changes in Zillow, missing values, alignment of the resampled interest rate); **trend** (wrong trend specification or detrended series not fully stationary); **model** (wrong order *p*, *q*, or linear/constant-coefficient assumption, structural breaks); **exogenous variable** (endogeneity, measurement error, or wrong lag structure); and **evaluation** (single train/test split, short test period, MSE sensitive to scale and outliers).

> **Highlight:** Biggest risks: **wrong trend or stationarity**, **wrong model order**, and **evaluation on a single short test period**.

---

## 7. Future prediction (how far ahead?)

Based on this housing dataset and our models, we’d be comfortable predicting only **a few months to about a year** ahead. Short-term one-step-ahead MSE was reasonable, but long-term multi-step forecasts without updating deteriorated quickly. Housing is affected by policy, credit shocks, and macro events that a simple AR/ARMAX doesn’t capture, and the further we extrapolate, the more trend and parameter uncertainty matter. We’d re-evaluate after a structural break (e.g. post-COVID) or in a different market; adding more exogenous variables or a regime-switching model might extend the plausible horizon.

> **Highlight:** We’d only be comfortable predicting **a few months to about a year** ahead; beyond that, forecasts degrade and structural factors dominate.

---

## Quick reference and things to mention


| Topic | Say this |
|-------|----------|
| Stationarity | We **tested** with ADF, didn’t assume; raw series non-stationary, detrended then re-tested. |
| Credit link | **corr(Δ price, rate) = −0.40, p ≈ 0** — significant; level of rate matters. |
| Trend | OLS linear detrend, validated with ADF on residuals. |
| Generalization | Plot → **ADF** → cross-correlation / **test significance of lags** → AR/ARMA or ARX/ARMAX → **AIC/BIC** and out-of-sample MSE. |
| Metrics | **MSE, RMSE, MAE** for accuracy; **AIC, BIC** for model selection. |
| ARX vs AR | ARX gave similar or slightly better **test MSE**; we’d use it when rate is available. |
| Horizon | Comfortable only **a few months to about a year** ahead. |
