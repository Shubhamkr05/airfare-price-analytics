# Airfare EDA Report

## Dataset Overview
- Rows: 6000
- Columns: 12
- Target: `fare_price`

## Data Quality
- origin: 0 missing
- destination: 0 missing
- cabin_class: 0 missing
- demand_index: 0 missing
- seasonality_index: 0 missing
- days_to_departure: 0 missing
- weekend_before_departure: 0 missing
- fuel_cost_index: 0 missing
- holiday_flag: 0 missing
- competitor_price: 0 missing
- load_factor: 0 missing
- fare_price: 0 missing

## Numeric Summary
```text
       demand_index  seasonality_index  days_to_departure  weekend_before_departure  fuel_cost_index  holiday_flag  competitor_price  load_factor  fare_price
count      6000.000           6000.000           6000.000                   6000.00         6000.000      6000.000          6000.000     6000.000    6000.000
mean          0.998              1.008             89.330                      0.36            1.001         0.124          8312.848        0.776   14171.466
std           0.251              0.200             51.488                      0.48            0.178         0.330          2100.500        0.114    4880.009
min           0.400              0.500              1.000                      0.00            0.600         0.000          2500.000        0.350    7244.620
25%           0.828              0.870             45.000                      0.00            0.878         0.000          6763.560        0.701   11192.642
50%           0.992              1.005             89.000                      0.00            0.999         0.000          8316.245        0.778   12844.170
75%           1.166              1.144            134.000                      1.00            1.121         0.000          9906.480        0.856   14915.702
max           1.800              1.700            179.000                      1.00            1.500         1.000         14579.960        0.980   37146.890
```

## Top Correlations With Fare Price
- competitor_price: 0.362
- demand_index: 0.110
- holiday_flag: 0.083
- seasonality_index: 0.065
- fuel_cost_index: 0.052
- weekend_before_departure: 0.050
- load_factor: 0.047
- days_to_departure: -0.044

## Outlier Counts (IQR Rule)
- holiday_flag: 745
- fare_price: 591
- load_factor: 41
- demand_index: 23
- fuel_cost_index: 22
- seasonality_index: 14
- days_to_departure: 0
- weekend_before_departure: 0
- competitor_price: 0

## Fare by Cabin Class
```text
                 count      mean    median       min       max
cabin_class                                                   
Business           580  26660.34  26795.72  16015.46  37146.89
Premium Economy   1038  16357.43  16525.46  10019.90  22270.96
Economy           4382  12000.64  12038.46   7244.62  17206.93
```

## Top 10 Expensive Routes
```text
                    count      mean    median
origin destination                           
DEL    BLR            778  16781.06  14630.30
BOM    CCU            743  16229.34  14222.85
DEL    CCU            766  15087.16  13027.29
       HYD            742  14908.99  12930.56
       BOM            739  14261.41  12380.43
BOM    BLR            761  12534.45  11172.75
       HYD            767  12146.50  10814.05
BLR    HYD            704  11223.34  10036.18
```

## Generated Charts
- `missing_values.png`
- `fare_distribution.png`
- `correlation_heatmap.png`
- `feature_vs_fare_scatter.png`
- `categorical_boxplots.png`