import psycopg2
import pandas as pd

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="mlops_db",
    user="postgres",
    password="Mohlala@2000"
)

# Query data
query = "SELECT * FROM customers"
df = pd.read_sql(query, conn)

print(df)

conn.close()