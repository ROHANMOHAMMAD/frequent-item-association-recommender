from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

DATA_FILE = "electronics_transactions_500.csv"

# -----------------------------
# 1. LOAD AND PREPARE DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)
products = [c for c in df.columns if c != "TransactionID"]
basket = df[products].eq("yes")

# -----------------------------
# 2. FREQUENT ITEMSET MINING
# -----------------------------
frequent_itemsets = apriori(
    basket,
    min_support=0.05,
    use_colnames=True
)

# -----------------------------
# 3. ASSOCIATION RULE MINING
# -----------------------------
rules = (
    association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.30
    )
    if not frequent_itemsets.empty
    else pd.DataFrame()
)

def item_text(items):
    return ", ".join(sorted(str(x).replace("_", " ") for x in items))

# -----------------------------
# 4. RECOMMENDATION ENGINE
# Only:
#   antecedent = selected product
#   consequent = ONE product
# This keeps recommendations clean and separate from bundles.
# -----------------------------
def get_recommendations(product, limit=6):
    if rules.empty:
        return []

    result = []

    for _, row in rules.iterrows():
        antecedents = set(row["antecedents"])
        consequents = set(row["consequents"])

        if antecedents == {product} and len(consequents) == 1:
            recommended = next(iter(consequents))

            result.append({
                "product": str(recommended).replace("_", " "),
                "support": round(float(row["support"]) * 100, 2),
                "confidence": round(float(row["confidence"]) * 100, 2),
                "lift": round(float(row["lift"]), 2)
            })

    # Best recommendation order: lift, then confidence, then support
    result.sort(
        key=lambda x: (x["lift"], x["confidence"], x["support"]),
        reverse=True
    )

    # Remove duplicate recommended products
    unique = []
    seen = set()
    for r in result:
        if r["product"] not in seen:
            unique.append(r)
            seen.add(r["product"])

    return unique[:limit]

# -----------------------------
# 5. PRODUCT BUNDLING ENGINE
# Multi-product frequent itemsets only.
# -----------------------------
def get_bundles(limit=12):
    if frequent_itemsets.empty:
        return []

    result = []

    for _, row in frequent_itemsets.iterrows():
        items = list(row["itemsets"])

        if len(items) >= 2:
            result.append({
                "products": item_text(items),
                "support": round(float(row["support"]) * 100, 2),
                "size": len(items)
            })

    # Prefer useful 2-3 product bundles and then support.
    result.sort(
        key=lambda x: (x["size"] if x["size"] <= 3 else 99, -x["support"])
    )

    return result[:limit]

# -----------------------------
# 6. DASHBOARD
# -----------------------------
@app.route("/")
def home():
    product_counts = (
        basket.sum()
        .sort_values(ascending=False)
        .head(8)
    )

    top_products = [
        {
            "name": str(name).replace("_", " "),
            "count": int(count),
            "percentage": round((count / len(df)) * 100, 1)
        }
        for name, count in product_counts.items()
    ]

    return render_template(
        "index.html",
        products=products,
        transactions=len(df),
        itemsets=len(frequent_itemsets),
        rules=len(rules),
        top_products=top_products
    )

# -----------------------------
# 7. RECOMMENDATION PAGE
# -----------------------------
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    selected = request.form.get("product", "")
    results = get_recommendations(selected) if selected else []

    return render_template(
        "recommend.html",
        products=products,
        selected=selected,
        results=results
    )

# -----------------------------
# 8. BUNDLE PAGE
# -----------------------------
@app.route("/bundles")
def bundles():
    return render_template(
        "bundles.html",
        bundles=get_bundles()
    )

# -----------------------------
# 9. ASSOCIATION RULES PAGE
# -----------------------------
@app.route("/rules")
def rules_page():
    result = []

    if not rules.empty:
        sorted_rules = rules.sort_values(
            ["lift", "confidence"],
            ascending=False
        ).head(30)

        for _, row in sorted_rules.iterrows():
            result.append({
                "antecedent": item_text(row["antecedents"]),
                "consequent": item_text(row["consequents"]),
                "support": round(float(row["support"]) * 100, 2),
                "confidence": round(float(row["confidence"]) * 100, 2),
                "lift": round(float(row["lift"]), 2)
            })

    return render_template(
        "rules.html",
        rules=result
    )

# -----------------------------
# 10. ANALYTICS PAGE
# -----------------------------
@app.route("/analytics")
def analytics():
    product_counts = (
        basket.sum()
        .sort_values(ascending=False)
        .head(10)
    )

    products_data = [
        {
            "name": str(name).replace("_", " "),
            "count": int(count),
            "percentage": round((count / len(df)) * 100, 1)
        }
        for name, count in product_counts.items()
    ]

    top_itemsets = []

    if not frequent_itemsets.empty:
        temp = frequent_itemsets.copy()
        temp["size"] = temp["itemsets"].apply(len)
        temp = temp[temp["size"] >= 2]
        temp = temp.sort_values("support", ascending=False).head(10)

        for _, row in temp.iterrows():
            top_itemsets.append({
                "items": item_text(row["itemsets"]),
                "support": round(float(row["support"]) * 100, 2),
                "size": int(row["size"])
            })

    return render_template(
        "analytics.html",
        products=products_data,
        itemsets=top_itemsets,
        transactions=len(df)
    )

if __name__ == "__main__":
    app.run(debug=True)
