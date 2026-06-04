# Coverage table

| adapter_family | task_family      | claim_type      | score_rows | seed_rows | report_rows | missing_pass_rows | successes | trials | pass_rate | wilson_low | wilson_high | median_rank | median_margin |
| -------------- | ---------------- | --------------- | ---------- | --------- | ----------- | ----------------- | --------- | ------ | --------- | ---------- | ----------- | ----------- | ------------- |
| ga2m_spline    | bilinear         | endpoints       | 41         | 41        | 2           | 0                 | 41        | 41     | 1.000     | 0.914      | 1.000       |             |               |
| ga2m_spline    | bilinear         | pair            | 41         | 41        | 2           | 0                 | 41        | 41     | 1.000     | 0.914      | 1.000       | 1.000       | 0.507         |
| ga2m_spline    | bilinear         | prediction      | 41         | 41        | 2           | 0                 | 41        | 41     | 1.000     | 0.914      | 1.000       |             |               |
| ga2m_spline    | bilinear         | support         | 41         | 41        | 2           | 0                 | 1         | 41     | 0.024     | 0.004      | 0.126       |             |               |
| ga2m_spline    | division_mixed   | endpoints       | 40         | 40        | 1           | 0                 | 1         | 40     | 0.025     | 0.004      | 0.129       |             |               |
| ga2m_spline    | division_mixed   | pair            | 40         | 40        | 1           | 0                 | 1         | 40     | 0.025     | 0.004      | 0.129       | inf         | 0.342         |
| ga2m_spline    | division_mixed   | prediction      | 40         | 40        | 1           | 0                 | 21        | 40     | 0.525     | 0.375      | 0.671       |             |               |
| ga2m_spline    | division_mixed   | support         | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       |             |               |
| ga2m_spline    | mixed_sparse     | endpoints       | 60         | 60        | 2           | 0                 | 1         | 60     | 0.017     | 0.003      | 0.089       |             |               |
| ga2m_spline    | mixed_sparse     | pair            | 60         | 60        | 2           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       | inf         | 0.370         |
| ga2m_spline    | mixed_sparse     | prediction      | 60         | 60        | 2           | 0                 | 28        | 60     | 0.467     | 0.346      | 0.591       |             |               |
| ga2m_spline    | mixed_sparse     | support         | 60         | 60        | 2           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       |             |               |
| ga2m_spline    | nested_trig      | endpoints       | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       |             |               |
| ga2m_spline    | nested_trig      | pair            | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       | inf         |               |
| ga2m_spline    | nested_trig      | prediction      | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       |             |               |
| ga2m_spline    | nested_trig      | support         | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       |             |               |
| ga2m_spline    | rational_product | endpoints       | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       |             |               |
| ga2m_spline    | rational_product | pair            | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       | inf         | 0.426         |
| ga2m_spline    | rational_product | prediction      | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       |             |               |
| ga2m_spline    | rational_product | support         | 20         | 20        | 1           | 0                 | 0         | 20     | 0.000     | 0.000      | 0.161       |             |               |
| ga2m_spline    | weak_centered    | endpoints       | 101        | 81        | 4           | 0                 | 100       | 101    | 0.990     | 0.946      | 0.998       |             |               |
| ga2m_spline    | weak_centered    | pair            | 101        | 81        | 4           | 0                 | 100       | 101    | 0.990     | 0.946      | 0.998       | 1.000       | 0.193         |
| ga2m_spline    | weak_centered    | prediction      | 101        | 81        | 4           | 0                 | 67        | 101    | 0.663     | 0.567      | 0.748       |             |               |
| ga2m_spline    | weak_centered    | support         | 101        | 81        | 4           | 0                 | 0         | 101    | 0.000     | 0.000      | 0.037       |             |               |
| gbm_hstat      | bilinear         | endpoints       | 62         | 60        | 1           | 0                 | 62        | 62     | 1.000     | 0.942      | 1.000       |             |               |
| gbm_hstat      | bilinear         | pair            | 62         | 60        | 1           | 0                 | 62        | 62     | 1.000     | 0.942      | 1.000       | 1.000       | 0.104         |
| gbm_hstat      | bilinear         | prediction      | 62         | 60        | 1           | 0                 | 0         | 62     | 0.000     | 0.000      | 0.058       |             |               |
| gbm_hstat      | bilinear         | support         | 62         | 60        | 1           | 0                 | 0         | 62     | 0.000     | 0.000      | 0.058       |             |               |
| gbm_hstat      | division_mixed   | endpoints       | 62         | 60        | 1           | 0                 | 0         | 62     | 0.000     | 0.000      | 0.058       |             |               |
| gbm_hstat      | division_mixed   | pair            | 62         | 60        | 1           | 0                 | 0         | 62     | 0.000     | 0.000      | 0.058       | inf         | 0.004         |
| gbm_hstat      | division_mixed   | prediction      | 62         | 60        | 1           | 0                 | 0         | 62     | 0.000     | 0.000      | 0.058       |             |               |
| gbm_hstat      | division_mixed   | support         | 62         | 60        | 1           | 0                 | 0         | 62     | 0.000     | 0.000      | 0.058       |             |               |
| gbm_hstat      | mixed_sparse     | endpoints       | 92         | 90        | 2           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat      | mixed_sparse     | pair            | 92         | 90        | 2           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       | inf         | 0.009         |
| gbm_hstat      | mixed_sparse     | prediction      | 92         | 90        | 2           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat      | mixed_sparse     | support         | 92         | 90        | 2           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat      | nested_trig      | endpoints       | 30         | 30        | 1           | 0                 | 0         | 30     | 0.000     | 0.000      | 0.114       |             |               |
| gbm_hstat      | nested_trig      | pair            | 30         | 30        | 1           | 0                 | 0         | 30     | 0.000     | 0.000      | 0.114       | inf         |               |
| gbm_hstat      | nested_trig      | prediction      | 30         | 30        | 1           | 0                 | 0         | 30     | 0.000     | 0.000      | 0.114       |             |               |
| gbm_hstat      | nested_trig      | support         | 30         | 30        | 1           | 0                 | 0         | 30     | 0.000     | 0.000      | 0.114       |             |               |
| gbm_hstat      | rational_product | endpoints       | 30         | 30        | 1           | 0                 | 2         | 30     | 0.067     | 0.018      | 0.213       |             |               |
| gbm_hstat      | rational_product | pair            | 30         | 30        | 1           | 0                 | 0         | 30     | 0.000     | 0.000      | 0.114       | inf         | 0.585         |
| gbm_hstat      | rational_product | prediction      | 30         | 30        | 1           | 0                 | 0         | 30     | 0.000     | 0.000      | 0.114       |             |               |
| gbm_hstat      | rational_product | support         | 30         | 30        | 1           | 0                 | 2         | 30     | 0.067     | 0.018      | 0.213       |             |               |
| gbm_hstat      | weak_centered    | endpoints       | 152        | 120       | 3           | 0                 | 93        | 152    | 0.612     | 0.533      | 0.686       |             |               |
| gbm_hstat      | weak_centered    | pair            | 152        | 120       | 3           | 0                 | 93        | 152    | 0.612     | 0.533      | 0.686       | 1.000       | 0.000         |
| gbm_hstat      | weak_centered    | prediction      | 152        | 120       | 3           | 0                 | 140       | 152    | 0.921     | 0.867      | 0.954       |             |               |
| gbm_hstat      | weak_centered    | support         | 152        | 120       | 3           | 0                 | 0         | 152    | 0.000     | 0.000      | 0.025       |             |               |
| pyKAN          | bilinear         | endpoints       | 896        | 752       | 4           | 0                 | 881       | 896    | 0.983     | 0.973      | 0.990       | 2.500       | 0.868         |
| pyKAN          | bilinear         | pair            | 1000       | 856       | 8           | 0                 | 980       | 1000   | 0.980     | 0.969      | 0.987       | 1.000       | 0.585         |
| pyKAN          | bilinear         | prediction      | 552        | 408       | 3           | 0                 | 506       | 552    | 0.917     | 0.891      | 0.937       |             |               |
| pyKAN          | bilinear         | support         | 448        | 376       | 2           | 0                 | 433       | 448    | 0.967     | 0.945      | 0.980       |             |               |
| pyKAN          | bilinear         | symbolic_status | 448        | 376       | 2           | 0                 | 448       | 448    | 1.000     | 0.991      | 1.000       |             |               |
| pyKAN          | division_mixed   | endpoints       | 560        | 400       | 2           | 0                 | 549       | 560    | 0.980     | 0.965      | 0.989       | 4.000       | 0.237         |
| pyKAN          | division_mixed   | pair            | 616        | 456       | 6           | 0                 | 62        | 616    | 0.101     | 0.079      | 0.127       | 2.000       | 0.293         |
| pyKAN          | division_mixed   | prediction      | 336        | 220       | 2           | 0                 | 320       | 336    | 0.952     | 0.924      | 0.970       |             |               |
| pyKAN          | division_mixed   | support         | 280        | 200       | 1           | 0                 | 272       | 280    | 0.971     | 0.945      | 0.985       |             |               |
| pyKAN          | division_mixed   | symbolic_status | 280        | 200       | 1           | 0                 | 280       | 280    | 1.000     | 0.986      | 1.000       |             |               |
| pyKAN          | exp_product      | endpoints       | 80         | 48        | 2           | 0                 | 79        | 80     | 0.988     | 0.933      | 0.998       | 2.000       | 0.978         |
| pyKAN          | exp_product      | pair            | 80         | 48        | 2           | 0                 | 79        | 80     | 0.988     | 0.933      | 0.998       | 1.000       | 0.603         |
| pyKAN          | exp_product      | prediction      | 40         | 24        | 1           | 0                 | 37        | 40     | 0.925     | 0.801      | 0.974       |             |               |
| pyKAN          | exp_product      | support         | 40         | 24        | 1           | 0                 | 39        | 40     | 0.975     | 0.871      | 0.996       |             |               |
| pyKAN          | exp_product      | symbolic_status | 40         | 24        | 1           | 0                 | 40        | 40     | 1.000     | 0.912      | 1.000       |             |               |
| pyKAN          | log_product      | endpoints       | 16         | 16        | 2           | 0                 | 16        | 16     | 1.000     | 0.806      | 1.000       | 2.000       | 0.989         |
| pyKAN          | log_product      | pair            | 16         | 16        | 2           | 0                 | 16        | 16     | 1.000     | 0.806      | 1.000       | 1.000       | 0.662         |
| pyKAN          | log_product      | prediction      | 8          | 8         | 1           | 0                 | 8         | 8      | 1.000     | 0.676      | 1.000       |             |               |
| pyKAN          | log_product      | support         | 8          | 8         | 1           | 0                 | 8         | 8      | 1.000     | 0.676      | 1.000       |             |               |
| pyKAN          | log_product      | symbolic_status | 8          | 8         | 1           | 0                 | 8         | 8      | 1.000     | 0.676      | 1.000       |             |               |
| pyKAN          | mixed_sparse     | endpoints       | 944        | 752       | 4           | 0                 | 126       | 944    | 0.133     | 0.113      | 0.157       | 33.500      | -0.012        |
| pyKAN          | mixed_sparse     | pair            | 1072       | 880       | 12          | 0                 | 926       | 1072   | 0.864     | 0.842      | 0.883       | 124.750     | 0.065         |
| pyKAN          | mixed_sparse     | prediction      | 600        | 420       | 4           | 0                 | 561       | 600    | 0.935     | 0.912      | 0.952       |             |               |
| pyKAN          | mixed_sparse     | support         | 472        | 376       | 2           | 0                 | 86        | 472    | 0.182     | 0.150      | 0.220       |             |               |
| pyKAN          | mixed_sparse     | symbolic_status | 472        | 376       | 2           | 0                 | 472       | 472    | 1.000     | 0.992      | 1.000       |             |               |
| pyKAN          | nested_trig      | endpoints       | 160        | 160       | 4           | 0                 | 86        | 160    | 0.537     | 0.460      | 0.613       | 32.500      | -0.179        |
| pyKAN          | nested_trig      | pair            | 216        | 216       | 8           | 0                 | 74        | 216    | 0.343     | 0.283      | 0.408       | 840.250     | -0.185        |
| pyKAN          | nested_trig      | prediction      | 136        | 100       | 3           | 0                 | 0         | 136    | 0.000     | 0.000      | 0.027       |             |               |
| pyKAN          | nested_trig      | support         | 80         | 80        | 2           | 0                 | 6         | 80     | 0.075     | 0.035      | 0.154       |             |               |
| pyKAN          | nested_trig      | symbolic_status | 80         | 80        | 2           | 0                 | 80        | 80     | 1.000     | 0.954      | 1.000       |             |               |
| pyKAN          | rational_product | endpoints       | 944        | 752       | 4           | 0                 | 607       | 944    | 0.643     | 0.612      | 0.673       | 8.250       | 0.005         |
| pyKAN          | rational_product | pair            | 1000       | 808       | 8           | 0                 | 961       | 1000   | 0.961     | 0.947      | 0.971       | 17.500      | 0.650         |

Showing first 80 of 219 rows.
