# Sampled Telco Churn Dataset

A lightweight test dataset sampled from the Telecom Churn benchmark for fast unit testing of feature generators.

## Dataset Overview

- **Source**: Customer Churn Prediction (Telecom)
- **Size**: ~1 MB total, 20,919 rows across all tables
- **Purpose**: Fast unit testing (<10s for 30 features vs ~30s for full dataset)
- **Coverage**: Time-series data, multi-table joins, all data types

## Files

### Main Tables
- `train.csv` - 700 rows
- `test.csv` - 300 rows

### Auxiliary Tables
- `billing_history_table.csv` - 11,823 rows
- `top_up_activation_history_table.csv` - 4,926 rows
- `customer_feedback_table.csv` - 3,170 rows

### Configuration & Ground Truth
- `problem.json` - Serialized ClassificationProblem configuration
- `ground_truth_features.json` - Reference feature names and descriptions from Telecom Churn benchmark

## Schema

### Main Table Columns
- `customer_id` (int) - Primary key
- `signup_date` (date) - Customer signup date
- `age` (int) - Customer age
- `gender` (string) - Customer gender
- `service_plan` (string) - Service plan type
- `contract_type` (string) - Contract type
- `churn_status` (int) - Target variable (0=retained, 1=churned)
- `reference_date` (date) - Reference date for temporal features

### Auxiliary Tables
Each auxiliary table has:
- `customer_id` - Foreign key to main table
- Date columns for temporal filtering
- Various event/transaction attributes

## Ground Truth Features

The dataset includes 4 ground truth features from the Telecom Churn benchmark:

1. **top_up_activity_decline_over_last_3_months** - Top up activity decline over last 3 months
2. **monthly_payment_timeliness** - Monthly payment timeliness
3. **negative_feedback_count_in_last_6_months** - Negative feedback count in last 6 months
4. **annual_increase_in_customer_support_calls** - Annual increase in customer support calls

These features are documented in `ground_truth_features.json` and can be used for comparison when testing feature generators.

## Usage

```python
async def test_my_generator(sampled_telco_churn, ctx):
    dataset = sampled_telco_churn
    
    # Tables already loaded in DuckDB
    train_table = dataset.train_table  # 'train'
    test_table = dataset.test_table    # 'test'
    problem = dataset.problem           # ClassificationProblem object
    
    # Auxiliary tables
    auxiliary_tables = dataset.auxiliary_tables
    # ('billing_history_table', 'top_up_activation_history_table', 'customer_feedback_table')
```

## Data Generation

This dataset was created using `create_golden_dataset.py` with:
- Simple random sampling to preserve natural class distribution (~80/20 not churned/churned)
- Filtering auxiliary tables to relevant customer IDs
- Random seed 42 for reproducibility
- Validation checks for data quality

## Performance

- **Load time**: <1s
- **Feature generation**: ~10s for 30 features (3x faster than full dataset)
- **Memory**: ~10 MB in DuckDB
