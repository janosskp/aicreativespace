import pandas as pd

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("csrd_esg_dataset.csv")

print("\n📊 INPUT DATA")
print(df.head())


# =========================================================
# 2. ESRS ONTOLOGY
# =========================================================
esg_concepts = {
    "energy_consumption": "energy kwh electricity power usage",
    "ghg_scope1": "fuel combustion direct emissions diesel gas co2",
    "ghg_scope2": "electricity indirect emissions grid power",
    "water_usage": "water liters consumption usage",
    "waste": "waste disposal kg production"
}

esrs_mapping = {
    "energy_consumption": "ESRS E1-5",
    "ghg_scope1": "ESRS E1-6",
    "ghg_scope2": "ESRS E1-6",
    "water_usage": "ESRS E3-4",
    "waste": "ESRS E5-3"
}


# =========================================================
# 3. INTELLIGENCE LAYER (RULE-BASED MVP)
# =========================================================
def semantic_map(column_name):
    col = column_name.lower()

    if "energy" in col or "kwh" in col or "power" in col:
        return "energy_consumption"

    if "scope1" in col or "co2" in col or "fuel" in col:
        return "ghg_scope1"

    if "electric" in col or "scope2" in col:
        return "ghg_scope2"

    if "water" in col:
        return "water_usage"

    if "waste" in col:
        return "waste"

    return "unknown"


# =========================================================
# 4. ESRS MAPPING
# =========================================================
def to_esrs(concept):
    return esrs_mapping.get(concept, "UNKNOWN")


# =========================================================
# 5. COLUMN ANALYSIS ENGINE
# =========================================================
def analyze_columns(df):
    results = []

    for col in df.columns:
        concept = semantic_map(col)
        esrs = to_esrs(concept)

        results.append({
            "column": col,
            "concept": concept,
            "esrs": esrs
        })

    return results


# =========================================================
# 6. FIXED NORMALIZATION (PANDAS 3.x SAFE)
# =========================================================
def normalize_dataframe(df):
    df = df.copy()

    for col in df.columns:
        # nur numerische Spalten anfassen
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].astype(float)

    return df


# =========================================================
# 7. COMPLIANCE ENGINE
# =========================================================
def check_compliance(mapped):
    required = ["ESRS E1-5", "ESRS E1-6"]

    present = list(set([m["esrs"] for m in mapped]))
    missing = [r for r in required if r not in present]

    return {
        "complete": len(missing) == 0,
        "missing": missing
    }


# =========================================================
# 8. REPORT GENERATOR
# =========================================================
def generate_report(mapped, compliance):
    report = "\n📄 CSRD / ESRS REPORT\n"
    report += "========================\n\n"

    for m in mapped:
        report += f"- {m['column']} → {m['concept']} → {m['esrs']}\n"

    report += "\n📊 Compliance Status:\n"

    if compliance["complete"]:
        report += "✔ All required ESRS metrics present\n"
    else:
        report += f"⚠ Missing: {compliance['missing']}\n"

    return report


# =========================================================
# 9. RUN PIPELINE
# =========================================================
df = normalize_dataframe(df)

mapped = analyze_columns(df)

compliance = check_compliance(mapped)

report = generate_report(mapped, compliance)


# =========================================================
# 10. OUTPUT
# =========================================================
print("\n🧠 MAPPING RESULTS:")
for m in mapped:
    print(m)

print("\n📊 COMPLIANCE:")
print(compliance)

print("\n📄 FINAL REPORT:")
print(report)


# =========================================================
# 11. EXPORT
# =========================================================
pd.DataFrame(mapped).to_csv("esrs_mapping_output.csv", index=False)

with open("csrd_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("\n✅ Export completed:")
print("- esrs_mapping_output.csv")
print("- csrd_report.txt")
