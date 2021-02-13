import pandas as pd
from google.cloud import storage
client = storage.Client()
df = pd.read_csv(r"gs://winerating-ml-marcelino-project/country_iso_data.csv")
