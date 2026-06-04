# Coverage table

| adapter_family  | task_family       | claim_type     | score_rows | seed_rows | report_rows | missing_pass_rows | successes | trials | pass_rate | wilson_low | wilson_high | median_rank | median_margin |
| --------------- | ----------------- | -------------- | ---------- | --------- | ----------- | ----------------- | --------- | ------ | --------- | ---------- | ----------- | ----------- | ------------- |
| epim_pairverify | bilinear          | candidate_pair | 870        | 210       | 1           | 0                 | 855       | 870    | 0.983     | 0.972      | 0.990       | 1.000       |               |
| epim_pairverify | bilinear          | endpoints      | 870        | 210       | 1           | 0                 | 853       | 870    | 0.980     | 0.969      | 0.988       | 1.000       |               |
| epim_pairverify | bilinear          | pair           | 1740       | 420       | 2           | 0                 | 1713      | 1740   | 0.984     | 0.978      | 0.989       | 1.000       | 0.557         |
| epim_pairverify | bilinear          | prediction     | 870        | 210       | 1           | 0                 | 833       | 870    | 0.957     | 0.942      | 0.969       |             |               |
| epim_pairverify | division_mixed    | candidate_pair | 870        | 210       | 1           | 0                 | 870       | 870    | 1.000     | 0.996      | 1.000       | 1.000       |               |
| epim_pairverify | division_mixed    | endpoints      | 870        | 210       | 1           | 0                 | 870       | 870    | 1.000     | 0.996      | 1.000       | 1.000       |               |
| epim_pairverify | division_mixed    | pair           | 1740       | 420       | 2           | 0                 | 102       | 1740   | 0.059     | 0.049      | 0.071       | 2.000       | -0.103        |
| epim_pairverify | division_mixed    | prediction     | 870        | 210       | 1           | 0                 | 831       | 870    | 0.955     | 0.939      | 0.967       |             |               |
| epim_pairverify | mixed_sparse      | candidate_pair | 1590       | 390       | 2           | 0                 | 1147      | 1590   | 0.721     | 0.699      | 0.743       | 28.000      |               |
| epim_pairverify | mixed_sparse      | endpoints      | 1590       | 390       | 2           | 0                 | 1011      | 1590   | 0.636     | 0.612      | 0.659       | 28.000      |               |
| epim_pairverify | mixed_sparse      | pair           | 3180       | 780       | 4           | 0                 | 2698      | 3180   | 0.848     | 0.836      | 0.860       | 1.000       | 0.239         |
| epim_pairverify | mixed_sparse      | prediction     | 1590       | 390       | 2           | 0                 | 1512      | 1590   | 0.951     | 0.939      | 0.961       |             |               |
| epim_pairverify | nested_trig       | candidate_pair | 1500       | 300       | 2           | 0                 | 91        | 1500   | 0.061     | 0.050      | 0.074       | 680.000     |               |
| epim_pairverify | nested_trig       | endpoints      | 1500       | 300       | 2           | 0                 | 65        | 1500   | 0.043     | 0.034      | 0.055       | 680.000     |               |
| epim_pairverify | nested_trig       | pair           | 3000       | 600       | 4           | 0                 | 0         | 3000   | 0.000     | 0.000      | 0.001       | 64.500      | -0.080        |
| epim_pairverify | nested_trig       | prediction     | 1500       | 300       | 2           | 0                 | 0         | 1500   | 0.000     | 0.000      | 0.003       |             |               |
| epim_pairverify | rational_product  | candidate_pair | 870        | 210       | 1           | 0                 | 870       | 870    | 1.000     | 0.996      | 1.000       | 1.000       |               |
| epim_pairverify | rational_product  | endpoints      | 870        | 210       | 1           | 0                 | 870       | 870    | 1.000     | 0.996      | 1.000       | 1.000       |               |
| epim_pairverify | rational_product  | pair           | 1740       | 420       | 2           | 0                 | 1740      | 1740   | 1.000     | 0.998      | 1.000       | 1.000       | 0.698         |
| epim_pairverify | rational_product  | prediction     | 870        | 210       | 1           | 0                 | 489       | 870    | 0.562     | 0.529      | 0.595       |             |               |
| epim_pairverify | sqrt_energy       | candidate_pair | 750        | 150       | 1           | 0                 | 750       | 750    | 1.000     | 0.995      | 1.000       | 1.000       |               |
| epim_pairverify | sqrt_energy       | endpoints      | 750        | 150       | 1           | 0                 | 750       | 750    | 1.000     | 0.995      | 1.000       | 1.000       |               |
| epim_pairverify | sqrt_energy       | pair           | 1500       | 300       | 2           | 0                 | 1500      | 1500   | 1.000     | 0.997      | 1.000       | 1.000       | 0.167         |
| epim_pairverify | sqrt_energy       | prediction     | 750        | 150       | 1           | 0                 | 750       | 750    | 1.000     | 0.995      | 1.000       |             |               |
| epim_pairverify | three_way_product | candidate_pair | 1500       | 300       | 2           | 0                 | 23        | 1500   | 0.015     | 0.010      | 0.023       | 1150.750    |               |
| epim_pairverify | three_way_product | endpoints      | 1500       | 300       | 2           | 0                 | 15        | 1500   | 0.010     | 0.006      | 0.016       | 1150.750    |               |
| epim_pairverify | three_way_product | pair           | 3000       | 600       | 4           | 0                 | 0         | 3000   | 0.000     | 0.000      | 0.001       | 71.000      | -0.059        |
| epim_pairverify | three_way_product | prediction     | 1500       | 300       | 2           | 0                 | 0         | 1500   | 0.000     | 0.000      | 0.003       |             |               |
| epim_pairverify | trig_product      | candidate_pair | 750        | 150       | 1           | 0                 | 739       | 750    | 0.985     | 0.974      | 0.992       | 1.000       |               |
| epim_pairverify | trig_product      | endpoints      | 750        | 150       | 1           | 0                 | 740       | 750    | 0.987     | 0.976      | 0.993       | 1.000       |               |
| epim_pairverify | trig_product      | pair           | 1500       | 300       | 2           | 0                 | 1479      | 1500   | 0.986     | 0.979      | 0.991       | 1.000       | 0.724         |
| epim_pairverify | trig_product      | prediction     | 750        | 150       | 1           | 0                 | 395       | 750    | 0.527     | 0.491      | 0.562       |             |               |
| epim_pairverify | weak_centered     | candidate_pair | 4560       | 750       | 5           | 0                 | 1757      | 4560   | 0.385     | 0.371      | 0.400       | 191.000     |               |
| epim_pairverify | weak_centered     | endpoints      | 4560       | 750       | 5           | 0                 | 2549      | 4560   | 0.559     | 0.545      | 0.573       | 191.000     |               |
| epim_pairverify | weak_centered     | pair           | 9120       | 1500      | 10          | 0                 | 4866      | 9120   | 0.534     | 0.523      | 0.544       | 1.000       | 0.037         |
| epim_pairverify | weak_centered     | prediction     | 4560       | 750       | 5           | 0                 | 3954      | 4560   | 0.867     | 0.857      | 0.877       |             |               |
| ga2m_spline     | bilinear          | endpoints      | 61         | 61        | 2           | 0                 | 61        | 61     | 1.000     | 0.941      | 1.000       |             |               |
| ga2m_spline     | bilinear          | pair           | 61         | 61        | 2           | 0                 | 61        | 61     | 1.000     | 0.941      | 1.000       | 1.000       | 0.510         |
| ga2m_spline     | bilinear          | prediction     | 61         | 61        | 2           | 0                 | 61        | 61     | 1.000     | 0.941      | 1.000       |             |               |
| ga2m_spline     | bilinear          | support        | 61         | 61        | 2           | 0                 | 1         | 61     | 0.016     | 0.003      | 0.087       |             |               |
| ga2m_spline     | division_mixed    | endpoints      | 60         | 60        | 1           | 0                 | 3         | 60     | 0.050     | 0.017      | 0.137       |             |               |
| ga2m_spline     | division_mixed    | pair           | 60         | 60        | 1           | 0                 | 3         | 60     | 0.050     | 0.017      | 0.137       | inf         | 0.342         |
| ga2m_spline     | division_mixed    | prediction     | 60         | 60        | 1           | 0                 | 23        | 60     | 0.383     | 0.271      | 0.510       |             |               |
| ga2m_spline     | division_mixed    | support        | 60         | 60        | 1           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       |             |               |
| ga2m_spline     | mixed_sparse      | endpoints      | 100        | 100       | 2           | 0                 | 3         | 100    | 0.030     | 0.010      | 0.085       |             |               |
| ga2m_spline     | mixed_sparse      | pair           | 100        | 100       | 2           | 0                 | 0         | 100    | 0.000     | 0.000      | 0.037       | inf         | 0.370         |
| ga2m_spline     | mixed_sparse      | prediction     | 100        | 100       | 2           | 0                 | 32        | 100    | 0.320     | 0.237      | 0.417       |             |               |
| ga2m_spline     | mixed_sparse      | support        | 100        | 100       | 2           | 0                 | 0         | 100    | 0.000     | 0.000      | 0.037       |             |               |
| ga2m_spline     | nested_trig       | endpoints      | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       |             |               |
| ga2m_spline     | nested_trig       | pair           | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       | inf         | -0.015        |
| ga2m_spline     | nested_trig       | prediction     | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       |             |               |
| ga2m_spline     | nested_trig       | support        | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       |             |               |
| ga2m_spline     | rational_product  | endpoints      | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       |             |               |
| ga2m_spline     | rational_product  | pair           | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       | inf         | 0.441         |
| ga2m_spline     | rational_product  | prediction     | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       |             |               |
| ga2m_spline     | rational_product  | support        | 40         | 40        | 1           | 0                 | 0         | 40     | 0.000     | 0.000      | 0.088       |             |               |
| ga2m_spline     | weak_centered     | endpoints      | 181        | 141       | 4           | 0                 | 179       | 181    | 0.989     | 0.961      | 0.997       |             |               |
| ga2m_spline     | weak_centered     | pair           | 181        | 141       | 4           | 0                 | 179       | 181    | 0.989     | 0.961      | 0.997       | 1.000       | 0.196         |
| ga2m_spline     | weak_centered     | prediction     | 181        | 141       | 4           | 0                 | 113       | 181    | 0.624     | 0.552      | 0.692       |             |               |
| ga2m_spline     | weak_centered     | support        | 181        | 141       | 4           | 0                 | 0         | 181    | 0.000     | 0.000      | 0.021       |             |               |
| gbm_hstat       | bilinear          | endpoints      | 92         | 90        | 1           | 0                 | 92        | 92     | 1.000     | 0.960      | 1.000       |             |               |
| gbm_hstat       | bilinear          | pair           | 92         | 90        | 1           | 0                 | 92        | 92     | 1.000     | 0.960      | 1.000       | 1.000       | 0.104         |
| gbm_hstat       | bilinear          | prediction     | 92         | 90        | 1           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat       | bilinear          | support        | 92         | 90        | 1           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat       | division_mixed    | endpoints      | 92         | 90        | 1           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat       | division_mixed    | pair           | 92         | 90        | 1           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       | inf         | 0.004         |
| gbm_hstat       | division_mixed    | prediction     | 92         | 90        | 1           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat       | division_mixed    | support        | 92         | 90        | 1           | 0                 | 0         | 92     | 0.000     | 0.000      | 0.040       |             |               |
| gbm_hstat       | mixed_sparse      | endpoints      | 152        | 150       | 2           | 0                 | 2         | 152    | 0.013     | 0.004      | 0.047       |             |               |
| gbm_hstat       | mixed_sparse      | pair           | 152        | 150       | 2           | 0                 | 0         | 152    | 0.000     | 0.000      | 0.025       | inf         | 0.009         |
| gbm_hstat       | mixed_sparse      | prediction     | 152        | 150       | 2           | 0                 | 0         | 152    | 0.000     | 0.000      | 0.025       |             |               |
| gbm_hstat       | mixed_sparse      | support        | 152        | 150       | 2           | 0                 | 2         | 152    | 0.013     | 0.004      | 0.047       |             |               |
| gbm_hstat       | nested_trig       | endpoints      | 60         | 60        | 1           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       |             |               |
| gbm_hstat       | nested_trig       | pair           | 60         | 60        | 1           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       | inf         | -0.000        |
| gbm_hstat       | nested_trig       | prediction     | 60         | 60        | 1           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       |             |               |
| gbm_hstat       | nested_trig       | support        | 60         | 60        | 1           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       |             |               |
| gbm_hstat       | rational_product  | endpoints      | 60         | 60        | 1           | 0                 | 4         | 60     | 0.067     | 0.026      | 0.159       |             |               |
| gbm_hstat       | rational_product  | pair           | 60         | 60        | 1           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       | inf         | 0.569         |
| gbm_hstat       | rational_product  | prediction     | 60         | 60        | 1           | 0                 | 0         | 60     | 0.000     | 0.000      | 0.060       |             |               |
| gbm_hstat       | rational_product  | support        | 60         | 60        | 1           | 0                 | 4         | 60     | 0.067     | 0.026      | 0.159       |             |               |

Showing first 80 of 255 rows.
