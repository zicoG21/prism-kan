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
