# Family-Level Overclaim Signature

Cells are weighted overclaim risk within an adapter family. Blank cells mean the family does not expose that transfer edge.

|                | Pred. -> pair   | Support -> pred.   | Candidate -> pair   | Symbolic -> expr.   | Fitted pair -> readout   | Fitted pair -> pruning   |
|:---------------|:----------------|:-------------------|:--------------------|:--------------------|:-------------------------|:-------------------------|
| pyKAN          | 16.2%           | 9.7%               |                     |                     | 25.6%                    | 20.8%                    |
| MLP-Hessian    | 13.0%           | 4.8%               |                     |                     |                          |                          |
| Symbolic/PySR  | 20.4%           | 8.8%               |                     | 83.4%               |                          |                          |
| Tree gates     | 66.6%           |                    | 54.3%               |                     |                          |                          |
| Sparse Lasso   | 38.1%           | 74.5%              |                     |                     |                          |                          |
| Spline Lasso   | 37.8%           | 27.0%              |                     |                     |                          |                          |
| Symbolic Lasso | 36.8%           | 8.4%               |                     | 100.0%              |                          |                          |
| Sparse library | 10.7%           | 36.1%              |                     |                     |                          |                          |
| Support screen |                 | 68.6%              |                     |                     |                          |                          |
| GA2M           | 17.8%           | 53.2%              |                     |                     |                          |                          |
| GBM-H          | 43.0%           | 63.6%              |                     |                     |                          |                          |
| EPIM verifier  | 15.6%           |                    | 17.0%               |                     |                          |                          |
