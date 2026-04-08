import pandas as pd

df = pd.read_csv("creditcard.csv")


"""
hier mit holen wir uns für jeden Datensatz die Klassifizierung,
ob es fraud ist oder nicht, damit das ML basierend auf den classes die features versteht:
"""

y = df["Class"].values
print(type(y))
print(y)

