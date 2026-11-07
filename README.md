# FMGCF-EWC: an operational, vulnerability-aware early warning chain for climate-related health risk

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24031500.svg)](https://doi.org/10.5281/zenodo.24031500) [![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
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

## Why it matters

Early warning systems save lives: disaster mortality is nearly six times lower in countries with comprehensive multi-hazard early warning systems than in countries with limited capabilities. The UN Early Warnings for All initiative aims to protect every person on Earth by an early warning system by 2027, yet 60% of countries report having a multi-hazard system and fewer than a third report capabilities for risk knowledge, the understanding of who is exposed and why, on which every other component depends ([WMO and UNDRR, 2025](https://wmo.int/news/media-centre/early-warning-systems-reach-new-heights-critical-gaps-jeopardize-global-progress)). Heat is a direct and growing threat in Southern Europe: an estimated 24,400 heat-related deaths occurred in 854 European cities in summer 2025, around 68% of them attributable to climate change, with Rome and Athens among the most affected capitals ([LSHTM and Imperial College London, 2025](https://www.lshtm.ac.uk/newsevents/news/2025/climate-change-driven-summer-heat-caused-16500-additional-deaths-across-europe)).

This chain addresses that gap in three ways: it places vulnerability at the centre of the decision to warn, so that the communities most at risk are warned first; it lets countries learn together without exchanging health data; and it measures, with confidence intervals, whether warnings actually reach the most vulnerable. Results on real data are produced by the workflow and will be reported here once reviewed.

## What it does

For every week of the warm season (ISO weeks 23 to 38) and every NUTS 2 region, the chain estimates the probability that deaths in the coming week will exceed the expected level by at least 10%, decides whether to issue a warning, and evaluates the warnings against what actually happened.

1. **Expected deaths and events.** For each region and week, expected deaths follow a linear trend fitted across the non-pandemic reference years (2015 to 2019, 2023 and 2024; deaths smoothed over three weeks), so that population ageing is not mistaken for excess mortality. The year being evaluated is always excluded, and the test year never enters the baseline of any other year. An event is a week with deaths at least 10% above expected.
2. **Regional graph.** Each region is linked to its nearest neighbours, across national borders, and its features are combined with theirs through one step of graph propagation (a simplified graph convolution, used here in place of the attention layer of the book to keep every computation inspectable).
3. **Vulnerability index.** The heat sensitivity of each region (the change in excess mortality per degree of temperature anomaly in the training years) is decomposed hierarchically into a country effect, estimated with partial pooling, and a regional component. The regional component is modelled by a Gaussian process with a Matérn 3/2 kernel over the share of people aged 65 and over and GDP per inhabitant, and mapped to an index between 0 and 1. The country effects are reported separately.
4. **Federated model.** One client per country trains locally; only model updates leave each country and are averaged with equal weight per country. A variant adds client-level differential privacy.
5. **Warnings and evaluation.** Calibrated probabilities are turned into warnings under three strategies, a single threshold, full vulnerability weighting and the weighting selected on the validation year, and all three are scored on a held-out year (2025) with region-level bootstrap confidence intervals.

## Operational extensions

**1. Validated vulnerability weighting of the warning threshold.** The book defines warnings through a calibrated risk threshold $`\theta`$ and, separately, a vulnerability index. The chain connects the two: the threshold for each region follows from the relative cost of a missed event and a false alarm, weighted by the region's vulnerability, so that the most vulnerable communities can be warned earlier. The strength of the weighting, $`\alpha \in [0, 1]`$, moves each region's weight between the average vulnerability $`\overline{\mathrm{MVI}}`$ and its own:

```math
\hat{p}_v \;\ge\; \tau_v(\alpha) + c_\alpha, \qquad \tau_v(\alpha) = \frac{C_{\text{FA}}}{C_{\text{FA}} + \left(\overline{\mathrm{MVI}} + \alpha\,(\mathrm{MVI}_v - \overline{\mathrm{MVI}})\right) C_{\text{miss}}}
```

The shift $`c_\alpha`$ keeps the number of warnings on the validation year equal for every $`\alpha`$, so that strategies differ only in where warnings go. A no-harm rule then selects $`\alpha`$: among the weightings whose overall probability of detection on the validation year is at least that of the single threshold ($`\alpha = 0`$), the one with the highest detection in the most vulnerable third of regions is chosen. If no weighting qualifies, the single threshold is kept and the decision is recorded.

**2. Fairness focused on missed warnings.** FairGLA introduces fairness regularisation and bias auditing for climate-health prediction. For an operational warning system, the fairness term is expressed on the warnings that matter most, those that are missed, by equalising the miss rate between a protected group of regions (below-median GDP per inhabitant) and the rest:

```math
\mathcal{L}_{\text{miss}} = \Big( \mathbb{E}\left[1-\hat{p}_v \mid y_v=1,\, s_v = 1\right] - \mathbb{E}\left[1-\hat{p}_v \mid y_v=1,\, s_v = 0\right] \Big)^{2}
```

**3. Client-level differential privacy on top of federated averaging.** The book's federated aggregation keeps raw data in each country and gives each participating country an equal voice. The chain keeps equal weighting and adds an optional differentially private variant, with each country's update clipped and perturbed before aggregation and the privacy cost accounted in zero-concentrated differential privacy:

```math
\theta^{t+1} = \theta^{t} + \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} \left( \mathrm{clip}\left(\Delta_i, C\right) + \mathcal{N}\left(0, \sigma^{2} C^{2} I\right) \right), \qquad \rho = \frac{T}{2|\mathcal{N}|\sigma^{2}}, \qquad \varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)}
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
| NUTS 2021 and NUTS 2024 region label points | Eurostat GISCO, via the `eurostat/Nuts2json` repository |

Training years are 2015 to 2019 and 2023, validation is 2024 and the test year is 2025. The pandemic years 2020 to 2022 are left out of training and of the mortality baseline. Where a country's regions were redrawn between NUTS versions (in this set, Portugal in NUTS 2024), the chain retrieves the series under each version and uses the one that covers every required year in the most regions. The version chosen for each country, every source call and every region whose deaths series was unavailable are listed in the **Data sources** sheet of the report.

`configs/synthetic.yaml` runs the same chain on simulated regions, so that the method can be inspected and tested without network access.

## Outputs

Each run writes `early_warning_report.xlsx` with the sheets Summary, Alert performance, Vulnerability weighting (validation results for every strength of weighting and the selection), Weighting vs uniform (differences with confidence intervals), By vulnerability tercile, By fairness group, Models (federated, unconstrained, differentially private and centralised variants with their AUC and privacy cost), Federated training, Vulnerability index (with each region's country effect and regional component), Test alerts (every region-week of the test year with its probability, thresholds and warnings under each strategy), Regional graph, Data coverage and Data sources.

## Running it on GitHub

Nothing needs to be run locally.

- **CI** runs the test suite on Python 3.10 to 3.13 on every push.
- **Early warning run** runs on every push to `main`, every Monday at 06:23 UTC and on demand. It runs the synthetic demonstration and the real-data chain, writes both summary tables to the run page and uploads the reports as the `early-warning-reports` artifact. Downloaded data are cached for the calendar month and refreshed at the start of the next.

## Results on synthetic data

With 40 simulated regions in four countries, the operational model reaches a test AUC of 0.893, and federated training matches centralised training on the same data (0.893 against 0.895). The no-harm rule selects a moderate weighting ($`\alpha = 0.5`$). Against the single threshold, it raises overall detection from 0.556 to 0.578 and lowers the false alarm ratio by 0.063 (95% CI 0.014 to 0.118), with a gain in detection in the most vulnerable third of regions of 0.045 (95% CI 0.000 to 0.156). With four national clients, client-level differential privacy at noise multiplier 1 gives ε ≈ 41 at δ = 10⁻⁵, a weak guarantee, and lowers test AUC from 0.893 to 0.858; with so few clients, strong privacy guarantees come at a substantial cost in accuracy, which the report makes explicit.

## Findings on real data

The first run on real data used an earlier configuration of the chain: expected deaths as a plain average of the reference years, a vulnerability index with latitude and longitude among its inputs, and full vulnerability weighting applied without a validation check. All 55 regions were retrieved, and the summer 2025 events peaked in ISO weeks 27 and 33 to 34, the early July and mid-August heatwaves, with 85 of the 163 events in Spain. The run produced three findings that shaped the present version:

1. **A vulnerability index learned from past mortality can anchor warnings to yesterday's risk.** With geographic inputs, the index placed every Greek region in the most vulnerable third and 14 of the 16 Spanish regions in the least vulnerable third, reflecting where heat-related excess mortality had been observed in the training years. Full weighting raised detection in the regions it ranked most vulnerable by 0.317 (95% CI 0.155 to 0.442), but moved warnings away from the Iberian Peninsula, where the 2025 heat struck hardest, and lowered overall detection from 0.583 to 0.331. The index now separates country effects from regional vulnerability, and weighting is applied only when it does not reduce overall detection on the validation year. Replayed on the 2025 predictions of that run, the no-harm rule would have kept the single threshold.
2. **Ageing populations inflate excess mortality measured against a flat baseline.** The share of event weeks rose from 13% in the training years to 31% in 2024, a drift consistent with a baseline that ignores the upward trend in deaths. Expected deaths now follow a regional linear trend.
3. **Regional boundaries change.** Three Portuguese regions had no 2025 series under their NUTS 2021 codes after the NUTS 2024 revision. The chain now chooses, for each country, the NUTS version with the most complete series and records the choice.

### Present version

The present version was run on the same four countries, with 57 regions retrieved and 52 regions carrying complete weekly series for training, validation (2024) and test (2025). The changes behave as intended, and the run leads to a clear operational conclusion.

| | First run | Present version |
|---|---|---|
| Test AUC, operational model | 0.654 | 0.728 |
| Event rate, training / 2024 / 2025 | 13% / 31% / 20% | 16% / 27% / 12% |
| Greek regions in the most vulnerable third | 13 of 13 | 7 of 13 |
| Spanish regions in the least vulnerable third | 14 of 16 | 5 of 16 |
| Selected vulnerability weighting | full, without validation | none ($`\alpha = 0`$) |

- **The trend baseline reduces the drift.** The share of event weeks in 2024 remains above the training years, which is consistent with the hot summer of 2024, but the gap is smaller, and 2025 now lies below the training rate. The 100 events of 2025 are spread across Spain (40), Italy (38), Greece (14) and Portugal (8).
- **The vulnerability index is no longer a country proxy.** The estimated country effects are small (between −0.007 and +0.005 on the heat-sensitivity scale), and every country has regions in all three vulnerability thirds. The index ranks regions within countries by age structure and income.
- **Vulnerability weighting moves detection rather than adding it.** With the index corrected, full weighting still raised detection in the most vulnerable third, from 0.533 to 0.733 (gain 0.200, 95% CI 0.082 to 0.348), but detection in the least vulnerable third fell from 0.556 to 0.139, and overall detection changed by −0.099 (95% CI −0.207 to 0.018). On the validation year every positive $`\alpha`$ lowered overall detection, so the no-harm rule kept the single threshold. With $`\alpha = 0`$, the chain detected 61 of the 100 events in 2025 (probability of detection 0.610, 95% CI 0.516 to 0.699) with a false alarm ratio of 0.745, the latter reflecting a cost ratio that treats a missed event as four times as costly as a false alarm.
- **Federated training costs nothing in accuracy here.** The federated model with the fairness penalty (test AUC 0.728) matched the centralised model (0.721). Client-level differential privacy lowered the test AUC to 0.677 at $`\varepsilon \approx 41`$, a weak formal guarantee with four clients; meaningful privacy would require many more participating authorities.

In operational terms, vulnerability weighting is a policy choice with an explicit price in this setting: protecting the most vulnerable regions more strongly costs detection elsewhere, and the chain reports that price instead of applying the weighting by default. An authority that accepts the trade-off can set the tolerance in the configuration; the report gives the expected gains and losses for each $`\alpha`$ on the validation year.

## Limitations

- The observed temperature of the target week stands in for a one-week-ahead forecast, so the measured skill is an upper bound on what an operational forecast would deliver.
- Excess mortality is used with a two-week reporting lag; national systems may report faster or slower.
- Eurostat publishes the new Portuguese NUTS 2024 regions only for recent years, and the former NUTS 2021 codes stop before 2025, so only two Portuguese regions (Norte and Algarve) have series long enough for training. Harmonising the series across the boundary revision would restore the rest of Portugal.
- Temperature is taken at one label point per region, which may not represent where people live in large or mountainous regions.
- Results describe association, not cause: weekly excess deaths in summer have causes other than heat.
- The chain supports, and does not replace, the judgement of public health authorities.

## Data attribution

Mortality, population and economic data: © European Union, Eurostat; reuse is authorised provided the source is acknowledged (Commission Decision 2011/833/EU). The data are aggregated to weekly NUTS 2 series and combined as described above. Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under CC BY 4.0; cite as Zippenfenig, P. (2023). Open-Meteo.com Weather API. https://doi.org/10.5281/zenodo.7970649. Region label points: © EuroGeographics for the administrative boundaries, Eurostat GISCO.

## Citation

Please cite both the software and the book; GitHub's **Cite this repository** button reads `CITATION.cff`.

> Sargiotis, D. (2026). *FMGCF-EWC: an operational, vulnerability-aware early warning chain for climate-related health risk* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24031500

> Sargiotis, D. (2027). *Blueprints for Future Living: AI-Driven Urban Innovation and Resilient Cities*. Springer Nature Switzerland. https://doi.org/10.1007/978-3-032-24178-8

## Licence

MIT. See `LICENSE`.
