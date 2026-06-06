# ClaimTransfer v1 standard-formula task cards

This generated registry maps a compact SRBench/Feynman-style subset into
ClaimTransfer task cards.  It is a public diagnostic wrapper, not a
hosted private leaderboard.

| task_id | family | support | pair claims | operators | purpose |
| --- | --- | ---: | ---: | --- | --- |
| `std_nguyen1_poly_d6_n2048` | standard_srbench_polynomial | 1 | 0 | plus,power | standard SR-style univariate polynomial; expression/operator recovery without pair claim |
| `std_nguyen5_trig_d6_n2048` | standard_srbench_trig | 1 | 0 | sin,cos,multiply,power,plus | standard SR-style trigonometric composition; symbolic status must not imply operator recovery |
| `std_keijzer_log_d6_n2048` | standard_srbench_log | 2 | 0 | log,plus,power | standard SR-style log/additive card; support and operator claims are separate |
| `std_bilinear_product_d8_n2048` | standard_pair_product | 3 | 1 | plus,multiply | aligned product positive control for pair and support claims |
| `std_feynman_energy_d8_n2048` | standard_feynman_style | 4 | 1 | plus,multiply,power | kinetic-plus-potential style card; pair claims are declared only for explicit bivariate term |
| `std_feynman_gravity_d8_n2048` | standard_feynman_style | 3 | 3 | multiply,divide,power,plus | gravity/Coulomb-style rational product; all declared pair claims are scorer-indexed |
| `std_ideal_gas_d8_n2048` | standard_feynman_style | 3 | 3 | multiply,divide,plus | ideal-gas-style ratio; support/pair/operator claims can split |
| `std_damped_wave_d8_n2048` | standard_feynman_style | 4 | 2 | multiply,exp,sin | damped-wave-style composition; pair evidence is scorer-sensitive |
| `std_harmonic_period_d6_n2048` | standard_feynman_style | 2 | 1 | sqrt,divide,plus | period-style sqrt ratio; operator and pair claims are distinct |
| `std_rosenbrock_d6_n2048` | standard_srbench_polynomial | 2 | 1 | plus,multiply,power | Rosenbrock-style nested polynomial; pair claim tests nonlinear coupling |
| `std_additive_no_pair_d8_n2048` | standard_negative_control | 3 | 0 | plus,power,sin | additive-only negative control; prediction/support claims must not imply pair recovery |
| `std_additive_exp_log_d8_n2048` | standard_negative_control | 3 | 0 | plus,exp,log,power,multiply | operator-rich additive card; symbolic/operator claims without pair claims |
| `std_two_products_d8_n2048` | standard_pair_product | 5 | 2 | plus,multiply | two independent products; pair claims require all declared pair rows |
| `std_trig_product_d8_n2048` | standard_pair_product | 3 | 1 | plus,multiply,sin,cos | trigonometric product card; prediction and pair evidence can split |
| `std_rational_plus_product_d8_n2048` | standard_rational_pair | 4 | 2 | plus,divide,power,multiply | rational plus product; support, partial-pair, and all-pair claims are separated |
| `std_exp_product_d8_n2048` | standard_compositional_pair | 3 | 1 | plus,exp,multiply | exponential product; expression status must not imply pair/operator recovery |
| `std_log_product_d8_n2048` | standard_compositional_pair | 3 | 1 | plus,log,multiply | log product with positive offset; pair and operator claims are distinct |
| `std_sqrt_product_d8_n2048` | standard_compositional_pair | 3 | 1 | plus,sqrt,multiply | sqrt product with positive offset; pair evidence and symbolic operator evidence split |
| `std_nested_pair_d8_n2048` | standard_compositional_pair | 4 | 1 | plus,sin,multiply | nested pair inside a sinusoid; pair scorer is a declared evidence object, not formula truth |
| `std_three_way_product_d8_n2048` | standard_higher_order | 4 | 1 | plus,multiply | three-way product stress card; pair claim is intentionally limited and scorer-indexed |
| `std_highdim_sparse_product_d12_n2048` | standard_pair_product | 4 | 1 | plus,multiply,power,sin | higher-dimensional sparse product; tests support budget and pair authorization |
| `std_feynman_coulomb_d8_n2048` | standard_feynman_style | 3 | 3 | multiply,divide,power,plus | Coulomb-style rational product; multiple pair claims are scorer-indexed |
| `std_feynman_ohm_d6_n2048` | standard_feynman_style | 2 | 1 | multiply | Ohm/power-style product; simple positive-control pair card |
| `std_feynman_lens_d6_n2048` | standard_feynman_style | 2 | 1 | divide,plus | thin-lens-style harmonic ratio; support/pair/operator claims can split |
| `std_feynman_frequency_d6_n2048` | standard_feynman_style | 3 | 3 | sqrt,multiply,divide,plus | frequency-style sqrt ratio; expression quality and pair claims are distinct |
| `std_nested_polynomial_pair_d8_n2048` | standard_srbench_polynomial | 4 | 1 | plus,multiply,power | nested polynomial pair; tests expression complexity versus pair recovery |
| `std_trig_mixed_two_pair_d8_n2048` | standard_compositional_pair | 4 | 2 | plus,sin,cos,multiply | two trigonometric products; all-pair claim is stricter than symbolic status |
| `std_division_pair_d8_n2048` | standard_rational_pair | 3 | 1 | plus,divide | division pair with additive nuisance; pair evidence and support evidence can split |
| `std_feynman_kepler_d6_n2048` | standard_feynman_style | 2 | 1 | sqrt,divide,power,plus | Kepler-style period relation; symbolic operator recovery is separate from pair recovery |
| `std_affine_no_pair_d12_n2048` | standard_negative_control | 3 | 0 | plus,multiply | sparse affine negative control; pair claims are illegal despite easy prediction |
