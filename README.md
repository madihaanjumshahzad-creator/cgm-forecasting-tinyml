# CGM Glucose Forecasting with Confidence Intervals and TinyML Deployment

Forecasting continuous glucose monitoring (CGM) data for type 2 diabetes, with a confidence interval for every prediction and an 8-bit model small enough for a microcontroller.

Related publication: time series analysis of continuous glucose monitoring data for type 2 diabetes, *PeerJ Computer Science* (2024).

## What the notebook does

**Part 1: Forecasting**
- Lag features (last 3 readings) plus month, day of week and hour
- SARIMA, Prophet and XGBoost compared on MAE, MSE, RMSE, MAPE and R²

**Part 2: Confidence and tiny-device deployment**
- 90% prediction intervals with split conformal prediction and XGBoost quantile regression, checked by test-set coverage
- A small neural network (401 parameters) with Monte Carlo dropout
- Full-integer (int8) quantization with TensorFlow Lite, and a check of whether the interval calibrated on the float model still holds after quantization
- Export of the int8 model for Edge Impulse (estimated latency, RAM and flash on a target MCU) and as a C header for TensorFlow Lite Micro, with the normalization values and interval half-width included

## Results (one patient, 15-minute CGM readings, last 25% of the data as test set)

**Forecasting (close to the published results)**

| Model | MAE | RMSE |
|---|---|---|
| XGBoost | 0.32 | 0.45 |
| SARIMA | 2.07 | 2.43 |
| Prophet | 2.22 | 2.64 |

**Confidence and tiny-device deployment**

| Model | Size | MAE | 90% interval coverage |
|---|---|---|---|
| XGBoost, 500 trees (conformal) | 2.7 MB | 0.40 | 84% |
| XGBoost quantile regression | | 0.31 | 89% |
| Neural network, float32 (conformal) | 3.5 KB | 0.40 | 81% |
| Neural network, int8 (conformal) | 3.5 KB | 0.40 | 80% |

- int8 quantization did not reduce accuracy or change interval coverage.
- Intervals calibrated on earlier months cover fewer than 90% of later readings, because the glucose pattern changes over time. A confidence estimate can become over-confident after deployment, so it needs to adapt.

## How to run

1. Open `CGM_forecasting_confidence_TinyML.ipynb` in Google Colab.
2. Upload your CGM file as `/content/cgm5.csv` with columns `Time` and `Glucose` (mmol/L).
3. Run all cells. The last cell downloads `cgm_mlp_int8.tflite` and `cgm_model_data.h`.

The patient data is not included in this repository.

## Author

Madiha Anjum, Lecturer, Computer Engineering, King Faisal University
[LinkedIn](https://www.linkedin.com/in/madihaanjum/) · [Google Scholar](https://scholar.google.com/citations?user=2HpTyoYAAAAJ)
