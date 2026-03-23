# Analyzing Results

```{objectives}
- Understand the sample results
- View accuracy landscapes and feature importance plots
```

## Sample Results

Sample results are included in the `results/` folder. These results demonstrate the effectiveness of various models and highlight key findings from the analysis, including accuracy landscapes and feature importance plots.

![Summary Results](/images/summary-results.png)

Filtered for 10-20 years summary results:

![Filtered summary](/images/image.png)

### Example Results by Target

**amo2 (10-20 years)**

![amo2 results](/images/image-3.png)

## Result File Format

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `target_feature` | Predicted variable |
| `max_lag` | Maximum lag in years |
| `corr_score` | Correlation coefficient |
| `mae_score` | Mean Absolute Error |
| `selected_features` | Features used by model |

## References

- Omrani, N. E., Keenlyside, N., Matthes, K., Boljka, L., Zanchettin, D., Jungclaus, J. H., & Lubis, S. W. (2022). Coupled stratosphere-troposphere-Atlantic multidecadal oscillation and its importance for near-future climate projection. NPJ Climate and Atmospheric Science, 5(1), 59.

```{keypoints}
- Results are saved to the `results/` folder
- Models are evaluated using correlation coefficients and MAE
- Accuracy landscapes show performance across different configurations
```
