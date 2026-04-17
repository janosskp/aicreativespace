import pandas as pd
from ai_function_layer import AIIntelligenceLayer

df = pd.read_csv("csrd_esg_dataset.csv")

ai = AIIntelligenceLayer()

mapped = ai.process_dataframe(df)

for m in mapped:
    print(m)