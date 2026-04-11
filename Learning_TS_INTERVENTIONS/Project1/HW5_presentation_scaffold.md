# Homework 5 — Presentation scaffold (dot points)

Use these as talking points; expand into full sentences for the 3–5 minute presentation and write-up.

---

## 1. Assumptions

- **Stationarity (for building a time series model)**
  - We needed a stationary series to fit AR/ARMA/ARMAX.
  - We did **not** assume stationarity; we **tested** it (e.g. ADF on raw series) and found the raw housing series **non-stationary** (e.g. trend, possibly unit root).
  - After detrending (see below), we tested again (e.g. ADF on residuals) and treated the detrended series as stationary enough to model.
  - *Why:* AR/ARMA theory assumes stationarity; we checked rather than assumed.

- **Linearity (implicit)**
  - We implicitly assumed a **linear** relationship between the lagged values of the series (and between the series and the exogenous variable in ARX/ARMAX). The model class is linear in the lags.

- **Model class**
  - We assumed the **detrended** series can be well approximated by an **AR**, **ARX**, or **ARMAX** process (linear, finite order).
  - This links to the course: housing prices (or their detrended version) modeled as ARMA/ARMAX with optional exogenous regressor (interest rate).

- **Domain / structural assumption: credit and housing**
  - **Assumption:** Housing prices are influenced by **credit conditions**; interest rates (e.g. 30-year mortgage) proxy for cost of credit, so they should have some **relationship** with house prices or their **changes**.
  - **How we tested it (Boston, MA, 2010–2017):**
    - **corr(Δ price, interest rate level) = −0.40, p ≈ 0.000** — statistically significant moderate negative correlation: higher mortgage rate is associated with lower month-over-month price growth.
    - **corr(Δ price, Δ interest rate) = 0.17, p = 0.11** — not significant at 5%; the *level* of the rate matters more than its month-to-month change for price growth in this sample.
    - We also compared AR vs ARX/ARMAX out-of-sample MSE to see if the rate adds predictive value.
  - We did not have to “assume” this blindly—we used interest rate as an exogenous variable and then **evaluated** whether including the rate in ARX/ARMAX improved forecasts.
- **What we did *not* assume (or assumed implicitly)**
  - No explicit distributional assumption (e.g. Gaussian errors) beyond what’s standard for ARIMA; we focused on second-order properties and MSE.
  - We treated the **aggregate series** (e.g. US or one region) as the object of interest; spatial heterogeneity or structural breaks are not explicitly modeled.

- **Summary for “Can you test all of these?”**
  - **Stationarity:** Yes — ADF (and optionally KPSS) before and after detrending.
  - **Model class (AR/ARMA):** Indirectly — AIC/BIC, ACF/PACF, and out-of-sample MSE.
  - **Credit/interest-rate link:** Yes — we computed corr(Δ price, rate) = −0.40 (p ≈ 0) and corr(Δ price, Δ rate) = 0.17 (p = 0.11), plus AR vs ARX/ARMAX comparison.

---

## 2. Trend removal

- **What we did:** Removed trend so the residual could be treated as stationary for AR/ARMA.
  - E.g. **OLS linear detrend:** regress series on time index, use **residuals** as the detrended series.
  - (You also tried polynomial detrend in the notebook; say which one you report.)
- **Other ways to remove trend:**
  - **Differencing** (e.g. first difference): \( y_t - y_{t-1} \); removes stochastic trend / unit root.
  - **Higher-order differencing** if needed.
  - **Polynomial (quadratic, etc.) or spline trend** then subtract.
- **Which we’d choose in the future and why:**
  - **Linear OLS detrend:** Simple, interpretable (constant growth rate), easy to re-add trend for forecasts; good when trend looks roughly linear.
  - **Differencing:** Good when we suspect unit root (random walk); no need to estimate trend; we’d use it if ADF suggests I(1) and we care about growth rates.
  - Mention that in your project you used OLS linear detrend (as required) and validated with ADF on residuals.

---

## 3. Generalization (new dataset, similar prediction task)

- **Steps we’d take:**
  - **Explore:** Plot series, check for missing values, definition of “prediction” (level vs. change, horizon).
  - **Stationarity:** Run a stationarity test — focus on the **Augmented Dickey–Fuller (ADF)** test; if non-stationary, detrend or difference, then re-test.
  - **Correlations and lag structure:** Run **basic correlations** (e.g. target vs. exogenous variable). From the course: look at **cross-correlation** between two signals (e.g. house price and interest rate), or the **power spectrum** / cross-spectrum, to see **which lags are meaningful**. Test the **statistical significance of lags** — correlations can be statistically tested, not just inspected for size, to see if a relationship is truly significant.
  - **Model class:** Start with AR (ACF/PACF), then ARMA/ARMAX if we have candidate exogenous variables.
  - **Order selection:** AIC/BIC and/or out-of-sample MSE (e.g. rolling one-step-ahead).
  - **Evaluate:** Hold out a test period; report MSE, RMSE, or MAE; compare models (e.g. AR vs ARX).
- **Which model and why:**
  - **AR/ARMA** when we only have the single time series.
  - **ARX/ARMAX** when we have a plausible exogenous driver (e.g. interest rate for housing, temperature for energy demand); use if it improves out-of-sample performance and is interpretable.
- **Does the type of dataset matter?**
  - Yes: frequency (daily vs monthly), presence of trend/seasonality, and whether exogenous variables are available and aligned in time.
- **Does domain knowledge come in?**
  - Yes: choice of exogenous variable (e.g. mortgage rate for housing), whether to model levels vs. growth rates, and how far ahead prediction is plausible (see Future prediction below).

---

## 4. Evaluation metrics

- **Prediction accuracy (point forecasts):** Focus on **MSE**, **RMSE**, and **MAE**.
  - **MSE** (mean squared error) — what we used; penalizes large errors more.
  - **RMSE** — same scale as the variable; easier to interpret.
  - **MAE** (mean absolute error) — robust to outliers.
- **Model fit / model selection:** **AIC** and **BIC** are very useful — they balance fit and complexity and are used for choosing \(p\), \(q\) (we compared in-sample MSE and AIC/BIC across orders).
- **How we used them:**
  - In-sample: MSE and **AIC/BIC** for AR (and ARX/ARMAX) order selection.
  - Out-of-sample: **MSE** (or RMSE/MAE) on held-out test period with short-term (one-step-ahead, rolling refit) strategy to compare AR vs ARX vs ARMAX.

---

## 5. Exogenous inputs (ARX vs AR only)

- **Comparison in our project:**
  - AR(p) only vs ARX(p) with interest rate (and ARMAX(p,q) with interest rate).
  - We compared **in-sample** and **out-of-sample MSE** (short-term prediction); e.g. ARX(p=4) had similar or slightly better test MSE than AR in your runs.
- **Why might ARX help or not?**
  - **Help:** If interest rates actually drive part of house price variation (credit channel), including them can reduce forecast error.
  - **Little or no help:** If the relationship is weak in-sample, or interest rate is already reflected in the recent lags of prices (collinearity with AR part), or the exogenous variable is not available at forecast time in practice.
- **Takeaway:** We found [summarize your result: e.g. “ARX gave modest improvement in test MSE” or “similar”; then say you’d still use ARX when interest rate is available and interpretable].

---

## 6. Error sources

- **Data:**
  - Measurement error or definition changes in Zillow series; missing data or alignment (e.g. interest rate resampled to monthly).
- **Stationarity / trend:**
  - Wrong trend specification (e.g. linear vs structural break); detrended series not fully stationary.
- **Model:**
  - Wrong order \((p,q)\); ARMA assumption (linear, constant coefficients) may be wrong; possible structural breaks in sample.
- **Exogenous variable:**
  - Interest rate may be endogenous or measured with error; lag structure or scaling might be wrong.
- **Evaluation:**
  - Single train/test split; short test period; MSE sensitive to scale and outliers.

---

## 7. Future prediction (how far ahead?)

- **Comfort horizon:** Based on this housing dataset and our models, we’d be comfortable predicting only **a few months to about a year** ahead (or whatever your conclusion is).
- **Why:**
  - Short-term (one-step-ahead) MSE was reasonable; long-term (multi-step without updating) blew up in your analysis — so forecasts degrade quickly.
  - Housing is affected by policy, credit shocks, and macro events that are hard to capture with a simple AR/ARMAX.
  - The more we extrapolate, the more trend and parameter uncertainty matter.
- **Caveats:** Would re-evaluate for a different market or after a structural break (e.g. post-COVID); might extend horizon if we added more exogenous variables or used a model that captures regime changes.

---

*Once you’re happy with these dot points, expand each into 2–3 sentences (or one short paragraph) for the actual presentation and Gradescope write-up.*
