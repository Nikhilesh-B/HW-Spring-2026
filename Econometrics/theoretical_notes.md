# Theoretical Understanding: Card & Krueger Table 4

## What Is Being Predicted?

The regressions in Table 4 predict **change in FTE employment** (ΔEmployment) at fast-food stores from February 1992 (before the NJ minimum wage increase) to November 1992 (after). Formally:

$$\Delta \text{Employment}_i = \beta_0 + \beta_1 \cdot \text{Policy Exposure}_i + \mathbf{X}_i'\boldsymbol{\gamma} + \varepsilon_i$$

where $\mathbf{X}_i$ are control variables (chain, ownership, region).

---

## Two Measures of Policy Exposure

### 1. New Jersey Dummy (Models i–ii)

**Variable:** $NJ_i = 1$ if store $i$ is in New Jersey, 0 if in Pennsylvania.

**Interpretation:** The coefficient $\beta_1$ is the *average difference* in employment change between NJ and PA stores. Under the **difference-in-differences** logic:
- PA stores experienced no minimum wage change → control group
- NJ stores experienced the $4.25 → $5.05 increase → treatment group
- **Parallel trends assumption:** In the absence of the policy, NJ and PA would have had the same employment growth. Any difference is attributed to the minimum wage.

**Result:** $\hat{\beta}_1 \approx 2.3$ → NJ stores gained about 2.3 more FTE workers than PA stores, on average. This is the *opposite* of the standard prediction (employment should fall).

---

### 2. Initial Wage Gap (Models iii–v)

**Variable:** 
$$\text{Gap}_i = \begin{cases} \frac{5.05 - \text{wage}_{\text{st},i}}{\text{wage}_{\text{st},i}} & \text{if NJ and } \text{wage}_{\text{st}} < 5.05 \\ 0 & \text{otherwise} \end{cases}$$

**Interpretation:** The gap measures the *proportional increase* in the starting wage needed to reach the new minimum. Examples:
- Store at $4.25 → gap = 0.19 (19% raise needed)
- Store at $5.00 → gap = 0.01 (1% raise)
- Store at $5.05+ or in PA → gap = 0 (no exposure)

**Causal logic:** *Within* New Jersey, stores with a larger gap were more exposed to the policy. If the competitive model holds, they should cut employment more. The coefficient $\beta_1$ gives the effect on ΔEmployment per unit increase in the gap (e.g., a 100% gap = doubling the wage).

**Result:** $\hat{\beta}_1 \approx 15.7$ (model iii) → a 10% larger gap is associated with about 1.6 more FTE workers. Again, no evidence of job loss; more-exposed stores added (or retained) more workers.

---

## Standard Theory vs. Card & Krueger

| Prediction | Standard competitive model | Card & Krueger estimates |
|------------|---------------------------|---------------------------|
| Effect of min wage on employment | Negative (firms cut jobs) | Zero or positive |
| NJ vs. PA | NJ should lose more jobs | NJ gains ~2.3 FTE relative to PA |
| Higher wage gap | More job loss | More employment (or no effect) |

**Possible explanations for the null/positive finding:**
1. **Monopsony:** Firms have market power in the labor market; a minimum wage can increase employment by raising the wage toward the competitive level.
2. **Efficiency wages:** Higher wages reduce turnover and raise productivity.
3. **Demand effects:** Higher wages → more spending by low-wage workers → more demand for fast food.
4. **Measurement/design:** Employment responses may be slow, or the sample may miss certain margins of adjustment.

---

## Role of Controls

- **Chain (bk, kfc, roys):** Different chains may have different employment practices; wendys is the reference.
- **Company-owned (co_owned):** Franchised vs. company-owned stores may respond differently.
- **Region (centralj, southj, pa1, pa2):** Local labor market conditions vary by geography.

Adding controls checks whether the main results are driven by *composition* (e.g., NJ having more of a certain chain). The coefficients remain similar, suggesting the findings are robust.

---

## Reduced-Form vs. Structural

Table 4 reports **reduced-form** estimates: they show the correlation between policy exposure and employment change, without specifying the full structural model (labor demand, labor supply, etc.). They answer: *What happened to employment when the minimum wage increased?* They do not directly identify elasticities or mechanisms, but they provide clean evidence on the sign and magnitude of the employment effect.
