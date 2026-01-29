# Triage Preview with Reason Codes (Logistic Regression)

Top 25 flagged items on TEST (sorted by band then score).

|   risk_score | decision_band   | reason_1   | reason_2   | reason_3   |   reason_1_score |   reason_2_score |   reason_3_score |   is_fraud |
|-------------:|:----------------|:-----------|:-----------|:-----------|-----------------:|-----------------:|-----------------:|-----------:|
|            1 | decline         | V14        | V12        | V4         |         15.9326  |          8.99718 |          5.34101 |          1 |
|            1 | decline         | V14        | V12        | V4         |         15.6005  |          8.68904 |          6.39347 |          1 |
|            1 | decline         | V14        | V12        | V4         |         15.3283  |          8.52552 |          6.37672 |          1 |
|            1 | decline         | V14        | V12        | V4         |         14.744   |          7.56389 |          6.30168 |          1 |
|            1 | decline         | Amount     | V20        | V4         |        196.211   |         50.0575  |         12.5708  |          0 |
|            1 | decline         | V14        | V12        | V4         |         15.0558  |          8.36194 |          6.36027 |          1 |
|            1 | decline         | V14        | V12        | V4         |         14.7832  |          8.19829 |          6.34408 |          1 |
|            1 | decline         | V14        | V12        | V4         |         14.5299  |          7.34286 |          6.37152 |          1 |
|            1 | decline         | V14        | V12        | V4         |         12.3755  |          6.85467 |          5.2525  |          1 |
|            1 | decline         | V14        | V12        | V4         |         13.9053  |          5.90508 |          5.86766 |          1 |
|            1 | decline         | V14        | V4         | V12        |         13.724   |          6.02419 |          5.60846 |          1 |
|            1 | decline         | V14        | V12        | V4         |         13.9112  |          6.29209 |          4.69519 |          1 |
|            1 | decline         | V14        | V4         | V12        |         13.576   |          6.28567 |          6.11175 |          1 |
|            1 | decline         | V14        | V4         | V12        |         13.45    |          6.00975 |          5.44438 |          1 |
|            1 | decline         | V14        | V12        | V4         |         11.7751  |          5.02735 |          4.90177 |          1 |
|            1 | decline         | V14        | V12        | V10        |          9.36788 |          5.78532 |          5.14621 |          1 |
|            1 | decline         | V14        | V12        | V10        |          9.55994 |          5.32539 |          4.08078 |          1 |
|            1 | decline         | V14        | V12        | V16        |          9.78359 |          5.93643 |          3.39424 |          1 |
|            1 | decline         | V14        | V12        | V4         |          9.37696 |          5.24837 |          4.79941 |          1 |
|            1 | decline         | V14        | V12        | V17        |          9.74317 |          5.7617  |          3.7404  |          1 |
|            1 | decline         | V14        | V4         | V12        |         11.3349  |          5.33395 |          4.49328 |          1 |
|            1 | decline         | V12        | V14        | V17        |          5.08414 |          4.78866 |          4.54936 |          1 |
|            1 | decline         | V14        | V4         | V12        |         11.0128  |          4.61666 |          4.34727 |          1 |
|            1 | decline         | V14        | V12        | V10        |          7.4434  |          5.22913 |          3.82925 |          1 |
|            1 | decline         | V14        | V4         | V12        |          9.19088 |          4.41433 |          3.60533 |          1 |

## Notes
- Reasons are the top positive logreg feature contributions for that row.
- Contribution = coefficient × (transformed feature value).
- Features are anonymized (V1–V28), so reasons are best interpreted as *signals* rather than human-meaningful fields.
