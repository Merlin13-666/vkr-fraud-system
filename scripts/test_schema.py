import pandas as pd
from src.fraud_system.io.schema import Schema

df = pd.DataFrame({
    "TransactionID": [1, 2],
    "TransactionDT": [100, 200],
    "isFraud": [0, 1],
})

schema = Schema("configs/schema_ieee_cis.yaml")
df2 = schema.apply(df)
print(df2.dtypes)
print(df2.columns.tolist())
print(df2.head())