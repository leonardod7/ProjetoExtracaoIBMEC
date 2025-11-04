#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Olist ETL & Analysis – IBM8915 (AP2)
------------------------------------
- Lê os CSV/XLS do dataset Olist
- Constrói uma tabela fato denormalizada + dimensões leves
- Cria a métrica delay_days (delivered - estimated)
- Salva saídas (CSV/Parquet quando possível)
- Compara 2 métodos (baseline vs regressão linear) e salva métricas (MAE etc.)

Execução:
  python build_olist_model.py \
      --data extracao/base_de_dados \
      --out  extracao/base_de_dados/outputs

Ou via variáveis de ambiente:
  OLIST_DATA_DIR=... OLIST_OUTPUT_DIR=... python build_olist_model.py
"""

import os
import sys
from typing import Optional, Dict
import pandas as pd


# --------------- Configuração (reprodutível) ----------------
DEFAULT_DATA_DIR = os.environ.get(
    "OLIST_DATA_DIR",
    os.path.join(os.getcwd(), "extracao", "base_de_dados")
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "OLIST_OUTPUT_DIR",
    os.path.join(DEFAULT_DATA_DIR, "outputs")
)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


# --------------- Helpers ----------------
def _read_any(path: str, **kwargs) -> pd.DataFrame:
    """
    Lê CSV ou Excel em DataFrame.
    - .csv: tenta ',' e depois ';'
    - .xlsx/.xls: usa read_excel
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)
        except Exception:
            return pd.read_csv(path, dtype=str, sep=";", low_memory=False, **kwargs)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        engine = "openpyxl" if lower.endswith(".xlsx") else None
        return pd.read_excel(path, dtype=str, engine=engine, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def _exists_any(data_dir: str, base: str) -> Optional[str]:
    """Procura {base}.csv/.xlsx/.xls em data_dir e retorna o caminho se existir."""
    candidates = [f"{base}.csv", f"{base}.CSV", f"{base}.xlsx", f"{base}.xls"]
    for c in candidates:
        p = os.path.join(data_dir, pbasename(c))
        if os.path.exists(p):
            return p
    return None


def pbasename(name: str) -> str:
    # evita barras acidentais em bases passadas
    return name.split("/")[-1].split("\\")[-1]


def load_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Carrega as tabelas necessárias do Olist (se existirem)."""
    file_map = {
        "orders": "olist_orders_dataset",
        "order_items": "olist_order_items_dataset",
        "order_payments": "olist_order_payments_dataset",
        "order_reviews": "olist_order_reviews_dataset",
        "customers": "olist_customers_dataset",
        "products": "olist_products_dataset",
        "sellers": "olist_sellers_dataset",
        "geolocation": "olist_geolocation_dataset",
        "category_translation": "product_category_name_translation",
    }

    dfs: Dict[str, pd.DataFrame] = {}
    for key, base in file_map.items():
        path = _exists_any(data_dir, base)
        if path is None:
            print(f"[WARN] Not found: {base} in {data_dir}")
            continue
        df = _read_any(path)
        df.columns = [c.strip().lower() for c in df.columns]
        dfs[key] = df
        print(f"[OK] Loaded {key:<20s} -> {df.shape} from {os.path.basename(path)}")
    return dfs


# --------------- Construção da fato ----------------
def build_denorm(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cria a tabela fato denormalizada seguindo o esquema definido."""
    required = ["orders", "order_items", "customers", "sellers", "products"]
    for r in required:
        if r not in dfs:
            raise RuntimeError(f"Missing required table: {r}")

    orders = dfs["orders"].copy()
    items = dfs["order_items"].copy()
    customers = dfs["customers"].copy()
    sellers = dfs["sellers"].copy()
    products = dfs["products"].copy()

    # Timestamps em datetime
    for col in [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # delay_days = delivered - estimated
    if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(orders.columns):
        orders["delay_days"] = (
            orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
        ).dt.days

    # Pagamentos agregados
    if "order_payments" in dfs:
        pays = dfs["order_payments"].copy()
        if "payment_value" in pays.columns:
            pays["payment_value"] = pd.to_numeric(pays["payment_value"], errors="coerce")
        agg_total = pays.groupby("order_id", as_index=False).agg(
            total_payment_value=("payment_value", "sum"),
            n_payments=("payment_sequential", "count")
        )
    else:
        agg_total = pd.DataFrame(columns=["order_id", "total_payment_value", "n_payments"])

    # Reviews agregados
    if "order_reviews" in dfs:
        rev = dfs["order_reviews"].copy()
        if "review_score" in rev.columns:
            rev["review_score"] = pd.to_numeric(rev["review_score"], errors="coerce")
        agg_rev = rev.groupby("order_id", as_index=False).agg(
            review_score=("review_score", "mean"),
            n_reviews=("review_score", "count"),
        )
    else:
        agg_rev = pd.DataFrame(columns=["order_id", "review_score", "n_reviews"])

    # Tradução de categorias (se existir)
    prod = products.copy()
    if "category_translation" in dfs and "product_category_name" in prod.columns:
        cat = dfs["category_translation"].copy()
        # procura coluna de inglês comum no arquivo de tradução
        trans_col = None
        for cand in [
            "product_category_name_english",
            "product_category_name_english ",
            "product_category_name_english\n"
        ]:
            if cand in cat.columns:
                trans_col = cand
                break
        if trans_col is None and len(cat.columns) >= 2:
            trans_col = cat.columns[1]
        cat = cat.rename(columns={trans_col: "product_category_name_english"})
        prod = prod.merge(
            cat[["product_category_name", "product_category_name_english"]].drop_duplicates(),
            on="product_category_name", how="left"
        )

    # Geolocalização agregada por prefixo CEP
    geo_agg = None
    if "geolocation" in dfs and {
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"
    }.issubset(set(dfs["geolocation"].columns)):
        geo = dfs["geolocation"][[
            "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"
        ]].copy()
        geo["geolocation_lat"] = pd.to_numeric(geo["geolocation_lat"], errors="coerce")
        geo["geolocation_lng"] = pd.to_numeric(geo["geolocation_lng"], errors="coerce")
        geo_agg = geo.groupby("geolocation_zip_code_prefix", as_index=False).agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
            n_points=("geolocation_lat", "count")
        )

    # Fato: uma linha por order_item
    fact = items.merge(orders, on="order_id", how="left", suffixes=("", "_order"))
    fact = fact.merge(prod, on="product_id", how="left", suffixes=("", "_product"))
    fact = fact.merge(sellers, on="seller_id", how="left", suffixes=("", "_seller"))
    fact = fact.merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
    fact = fact.merge(agg_total, on="order_id", how="left")
    fact = fact.merge(agg_rev, on="order_id", how="left")

    # Geo customer/seller
    if geo_agg is not None:
        if "customer_zip_code_prefix" in fact.columns:
            fact = fact.merge(
                geo_agg.rename(columns={
                    "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                    "lat": "customer_lat", "lng": "customer_lng", "n_points": "customer_zip_points"
                }),
                on="customer_zip_code_prefix", how="left"
            )
        if "seller_zip_code_prefix" in fact.columns:
            fact = fact.merge(
                geo_agg.rename(columns={
                    "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                    "lat": "seller_lat", "lng": "seller_lng", "n_points": "seller_zip_points"
                }),
                on="seller_zip_code_prefix", how="left"
            )

    # Numéricos
    for c in ["price", "freight_value"]:
        if c in fact.columns:
            fact[c] = pd.to_numeric(fact[c], errors="coerce")

    # KPI: GMV do item
    if {"price", "freight_value"}.issubset(fact.columns):
        fact["item_gmv"] = fact["price"].fillna(0) + fact["freight_value"].fillna(0)

    # Colunas em ordem amigável (se existirem)
    preferred_cols = [
        # Keys
        "order_id", "order_item_id", "product_id", "seller_id", "customer_id",
        # Order
        "order_status", "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "delay_days",
        # Item values
        "price", "freight_value", "item_gmv", "shipping_limit_date",
        # Customer dims
        "customer_city", "customer_state", "customer_zip_code_prefix",
        "customer_lat", "customer_lng",
        # Seller dims
        "seller_city", "seller_state", "seller_zip_code_prefix",
        "seller_lat", "seller_lng",
        # Product dims
        "product_category_name", "product_category_name_english",
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
        # Payments & reviews
        "total_payment_value", "n_payments", "review_score", "n_reviews",
    ]
    cols_existing = [c for c in preferred_cols if c in fact.columns]
    fact = fact[cols_existing + [c for c in fact.columns if c not in cols_existing]]

    return fact


# --------------- Saídas ----------------
def save_outputs(fact: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """Salva fato e dimensões em CSV (e Parquet se disponível)."""
    paths: Dict[str, str] = {}

    fact_csv = os.path.join(output_dir, "olist_order_lines_denorm.csv")
    fact.to_csv(fact_csv, index=False)
    paths["fact_csv"] = fact_csv

    try:
        fact_parquet = os.path.join(output_dir, "olist_order_lines_denorm.parquet")
        fact.to_parquet(fact_parquet, index=False)
        paths["fact_parquet"] = fact_parquet
    except Exception as e:
        print(f"[INFO] Skipping Parquet (pyarrow/fastparquet not installed?): {e}")

    # Dimensões leves
    if "product_id" in fact.columns:
        cols = [c for c in ["product_id", "product_category_name", "product_category_name_english"] if c in fact.columns]
        dim_prod = fact[cols].drop_duplicates()
        p = os.path.join(output_dir, "dim_products.csv")
        dim_prod.to_csv(p, index=False); paths["dim_products"] = p

    if {"seller_id", "seller_city", "seller_state"}.issubset(fact.columns):
        dim_sell = fact[["seller_id", "seller_city", "seller_state"]].drop_duplicates()
        p = os.path.join(output_dir, "dim_sellers.csv")
        dim_sell.to_csv(p, index=False); paths["dim_sellers"] = p

    if {"customer_id", "customer_city", "customer_state"}.issubset(fact.columns):
        dim_cust = fact[["customer_id", "customer_city", "customer_state"]].drop_duplicates()
        p = os.path.join(output_dir, "dim_customers.csv")
        dim_cust.to_csv(p, index=False); paths["dim_customers"] = p

    return paths


# --------------- Avaliação (2 métodos) ----------------
def evaluate_two_methods(fact: pd.DataFrame, out_dir: str) -> None:
    """
    Compara:
      (i) baseline = média
      (ii) regressão linear (review_score ~ delay_days)
    Salva métricas em CSV: outputs/metrics_model_comparison.csv
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    needed = {"delay_days", "review_score"}
    if not needed.issubset(fact.columns):
        print("[INFO] PULANDO avaliação: faltam colunas delay_days/review_score.")
        return

    dfm = fact[list(needed)].dropna()
    if len(dfm) < 1000:
        print("[INFO] PULANDO avaliação: amostra pequena (<1000).")
        return

    X = dfm[["delay_days"]].astype(float).values
    y = dfm["review_score"].astype(float).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline_pred = np.full_like(yte, fill_value=ytr.mean(), dtype=float)
    baseline_mae = float(mean_absolute_error(yte, baseline_pred))

    lr = LinearRegression().fit(Xtr, ytr)
    yhat = lr.predict(Xte)
    linear_mae = float(mean_absolute_error(yte, yhat))
    coef = float(lr.coef_[0]); intercept = float(lr.intercept_)

    corr = float(pd.Series(dfm["delay_days"]).corr(pd.Series(dfm["review_score"])))

    metrics = pd.Series({
        "n_obs": len(dfm),
        "corr_delay_score": corr,
        "baseline_mae": baseline_mae,
        "linear_mae": linear_mae,
        "linear_coef_delay": coef,
        "linear_intercept": intercept,
    })
    metrics_path = os.path.join(out_dir, "metrics_model_comparison.csv")
    metrics.to_csv(metrics_path)
    print(f"[OK] Métricas salvas em: {metrics_path}")


# --------------- Main ----------------
def run(data_dir: str, output_dir: str) -> None:
    print(f"[INFO] Reading input files from: {data_dir}")
    dfs = load_tables(data_dir)
    fact = build_denorm(dfs)
    print(f"[OK] Denormalized order lines shape: {fact.shape}")
    outputs = save_outputs(fact, output_dir)
    evaluate_two_methods(fact, output_dir)
    print("[OK] Files written:")
    for k, v in outputs.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build Olist denormalized fact table and evaluate simple models.")
    ap.add_argument("--data", default=DEFAULT_DATA_DIR, help="Diretório com os CSV/XLS do Olist")
    ap.add_argument("--out",  default=DEFAULT_OUTPUT_DIR, help="Diretório de saída (outputs)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run(args.data, args.out)
