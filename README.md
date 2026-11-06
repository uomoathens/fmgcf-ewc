# FMGCF-EWC: an operational, vulnerability-aware early warning chain for climate-related health risk

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Early warning run](../../actions/workflows/early-warning.yml/badge.svg)](../../actions/workflows/early-warning.yml)

**Dimitrios Sargiotis** · School of Civil Engineering, National Technical University of Athens · [ORCID 0009-0006-3644-4561](https://orcid.org/0009-0006-3644-4561) · [users.ntua.gr/dims](https://users.ntua.gr/dims/)

This repository develops three frameworks introduced in

> Sargiotis, D. (2027). *Blueprints for Future Living: AI-Driven Urban Innovation and Resilient Cities*. Springer Nature Switzerland. Chapter 6. https://doi.org/10.1007/978-3-032-24178-8

into one runnable early warning chain, tested on real European data:

| Framework in the book | Role in the chain | Module |
|---|---|---|
| AI-Driven Early Warning Coordination for Global Climate-Health Resilience (FMGCF-EWC), §6.3.3 | Regional warning scores, cross-border graph, federated learning across countries | `graph.py`, `federated.py`, `pipeline.py` |
| Multifactorial Vulnerability Index with Hierarchical Gaussian Process (MVI-HGP), §6.2 | Who is most at risk, with uncertainty | `vulnerability.py` |
| Fairness-Constrained Graph Learning Architecture (FairGLA), §6.3.6 | Equitable warnings across population groups | `federated.py`, `alerts.py` |

The book sets out the architecture and its mathematical core: graph-based warning scores over regional nodes, federated aggregation across jurisdictions, a Gaussian-process vulnerability index with a Matérn 3/2 kernel, and fairness-aware learning with bias auditing. Building on that foundation, this repository joins the three frameworks into a single decision chain and extends them in four operational directions, described below.

## What it does

For every week of the warm season (ISO weeks 23 to 38) and every NUTS 2 region, the chain estimates the probability that deaths in the coming week will exceed the expected level by at least 10%, decides whether to issue a warning, and evaluates the warnings against what actually happened.

1. **Expected deaths and events.** Expected weekly deaths are the mean of the same week (smoothed over three weeks) in the non-pandemic reference years 2015 to 2019 and 2023 to 2025, always excluding the year being evaluated. An event is a week with deaths at least 10% above that baseline.
2. **Regional graph.** Each region is linked to its nearest neighbours, across national borders, and its features are combined with theirs through one step of graph propagation (a simplified graph convolution, used here in place of the attention layer of the book to keep every computation inspectable).
3. **Vulnerability index.** The heat sensitivity of each region (the change in excess mortality per degree of temperature anomaly in the training years) is smoothed by a Gaussian process with a Matérn 3/2 kernel over the share of people aged 65 and over, GDP per inhabitant, latitude and longitude, and mapped to an index between 0 and 1.
4. **Federated model.** One client per country trains locally; only model updates leave each country and are averaged with equal weight per country. A variant adds client-level differential privacy.
5. **Warnings and evaluation.** Calibrated probabilities are turned into warnings, which are scored on a held-out year (2025) with region-level bootstrap confidence intervals.

## Operational extensions

**1. Vulnerability-aware warning threshold.** The book defines warnings through a calibrated risk threshold $`\theta`$ and, separately, a vulnerability index. The chain connects the two: the threshold for each region follows from the relative cost of a missed event and a false alarm, scaled by the region's vulnerability, so that the most vulnerable communities are warned earlier.

```math
\hat{p}_v \;\ge\; \tau_v = \frac{C_{\text{FA}}}{C_{\text{FA}} + \mathrm{MVI}_v \, C_{\text{miss}}}
```

To isolate the effect of where warnings go from how many are issued, it is compared with a single threshold set to produce exactly the same number of warnings on the validation year.

**2. Fairness focused on missed warnings.** FairGLA introduces fairness regularisation and bias auditing for climate-health prediction. For an operational warning system, the fairness term is expressed on the warnings that matter most, those that are missed, by equalising the miss rate between a protected group of regions (below-median GDP per inhabitant) and the rest:

```math
\mathcal{L}_{\text{miss}} = \Big( \mathbb{E}\left[1-\hat{p}_v \mid y_v=1,\, s_v = 1\right] - \mathbb{E}\left[1-\hat{p}_v \mid y_v=1,\, s_v = 0\right] \Big)^{2}
```

**3. Client-level differential privacy on top of federated averaging.** The book's federated aggregation keeps raw data in each country and gives each participating country an equal voice. The chain keeps equal weighting and adds an optional differentially private variant, with each country's update clipped and perturbed before aggregation and the privacy cost accounted in zero-concentrated differential privacy:

```math
\theta^{t+1} = \theta^{t} + \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} \left( \operatorname{clip}\left(\Delta_i, C\right) + \mathcal{N}\left(0, \sigma^{2} C^{2} I\right) \right), \qquad \rho = \frac{T}{2|\mathcal{N}|\sigma^{2}}, \qquad \varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)}
```

**4. An evaluation layer.** Every run reports the standard verification scores of warning systems, overall, by vulnerability tercile and by fairness group, with 95% confidence intervals from resampling regions:

```math
\text{POD} = \frac{\text{hits}}{\text{hits} + \text{misses}}, \qquad \text{FAR} = \frac{\text{false alarms}}{\text{hits} + \text{false alarms}}, \qquad \text{CSI} = \frac{\text{hits}}{\text{hits} + \text{misses} + \text{false alarms}}
```

## Data

The real-data configuration `configs/europe_south.yaml` covers the NUTS 2 regions of Greece, Italy, Spain and Portugal, excluding the outermost regions (Canarias, Açores, Madeira) and the autonomous cities of Ceuta and Melilla, downloaded on the first run and cached:

| Data | Source |
|---|---|
| Weekly deaths by NUTS region | Eurostat, `demo_r_mwk3_ts`; a NUTS 2 series is rebuilt from its NUTS 3 regions when needed, only for weeks in which every NUTS 3 region reports |
| Share of population aged 65 and over | Eurostat, `demo_r_pjanind2` (`PC_Y65_MAX`), most recent available year |
| GDP per inhabitant in PPS | Eurostat, `nama_10r_2gdp` (`PPS_EU27_2020_HAB`), most recent available year |
| Daily maximum temperature at each region's label point | Open-Meteo historical weather API (ERA5-based) |
| NUTS 2021 region label points | Eurostat GISCO, via the `eurostat/Nuts2json` repository |

Training years are 2015 to 2019 and 2023, validation is 2024 and the test year is 2025. The pandemic years 2020 to 2022 are left out of training and of the mortality baseline. Every source call, and every region whose deaths series was unavailable, is listed in the **Data sources** sheet of the report.

`configs/synthetic.yaml` runs the same chain on simulated regions, so that the method can be inspected and tested without network access.

## Outputs

Each run writes `early_warning_report.xlsx` with the sheets Summary, Alert performance, Vulnerability-aware gain, By vulnerability tercile, By fairness group, Models (federated, unconstrained, differentially private and centralised variants with their AUC and privacy cost), Federated training, Vulnerability index, Test alerts (every region-week of the test year with its probability, threshold and warnings), Regional graph, Data coverage and Data sources.

## Running it on GitHub

Nothing needs to be run locally.

- **CI** runs the test suite on Python 3.10 to 3.13 on every push.
- **Early warning run** runs on every push to `main`, every Monday at 06:23 UTC and on demand. It runs the synthetic demonstration and the real-data chain, writes both summary tables to the run page and uploads the reports as the `early-warning-reports` artifact. Downloaded data are cached for the calendar month and refreshed at the start of the next.

## Results on synthetic data

With 40 simulated regions in four countries, the operational model reaches a test AUC of 0.879. At the same number of warnings, the vulnerability-aware threshold raises the probability of detection in the most vulnerable third of regions by 0.102 (95% CI 0.000 to 0.191), at the cost of a slightly higher false alarm ratio overall (0.286 against 0.262). With four national clients, client-level differential privacy at noise multiplier 1 gives ε ≈ 41 at δ = 10⁻⁵, a weak guarantee, and lowers AUC from 0.879 to 0.860 on the test year and from 0.917 to 0.785 on the validation year; with so few clients, strong privacy guarantees come at a substantial cost in accuracy, which the report makes explicit. Federated training matches centralised training on the same data (test AUC 0.879 against 0.873). Results on real data are produced by the GitHub workflow.

## Limitations

- The observed temperature of the target week stands in for a one-week-ahead forecast, so the measured skill is an upper bound on what an operational forecast would deliver.
- Excess mortality is used with a two-week reporting lag; national systems may report faster or slower.
- Temperature is taken at one label point per region, which may not represent where people live in large or mountainous regions.
- Results describe association, not cause: weekly excess deaths in summer have causes other than heat.
- The chain supports, and does not replace, the judgement of public health authorities.

## Data attribution

Mortality, population and economic data: © European Union, Eurostat; reuse is authorised provided the source is acknowledged (Commission Decision 2011/833/EU). The data are aggregated to weekly NUTS 2 series and combined as described above. Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under CC BY 4.0; cite as Zippenfenig, P. (2023). Open-Meteo.com Weather API. https://doi.org/10.5281/zenodo.7970649. Region label points: © EuroGeographics for the administrative boundaries, Eurostat GISCO.

## Citation

Please cite both the software and the book; GitHub's **Cite this repository** button reads `CITATION.cff`.

> Sargiotis, D. (2026). *FMGCF-EWC: an operational, vulnerability-aware early warning chain for climate-related health risk* (Version 1.0.0) [Computer software]. https://github.com/uomoathens/fmgcf-ewc

> Sargiotis, D. (2027). *Blueprints for Future Living: AI-Driven Urban Innovation and Resilient Cities*. Springer Nature Switzerland. https://doi.org/10.1007/978-3-032-24178-8

## Licence

MIT. See `LICENSE`.
