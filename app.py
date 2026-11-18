"""App de análisis de ventas con Machine Learning.

- Lee un CSV de ventas (fecha, producto, cantidad, precio_unitario[, categoria])
- Clasifica la rotación de productos con K-Means (Alta / Media / Baja)
- Predice las unidades del próximo mes con un Random Forest
- Muestra gráficos interactivos

Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Ventas + ML", page_icon="📦", layout="wide")

COLORES_ROTACION = {"Alta": "#2a9d8f", "Media": "#e9c46a", "Baja": "#e76f51"}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
COLUMNAS_REQUERIDAS = {"fecha", "producto", "cantidad", "precio_unitario"}


# ---------------------------------------------------------------- datos
@st.cache_data
def cargar_datos(archivo) -> pd.DataFrame:
    df = pd.read_csv(archivo)
    df.columns = [c.strip().lower() for c in df.columns]
    faltan = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(sorted(faltan))}")
    df["fecha"] = pd.to_datetime(df["fecha"])
    if "categoria" not in df.columns:
        df["categoria"] = "General"
    df["total"] = df["cantidad"] * df["precio_unitario"]
    df["mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    return df


def ventas_mensuales(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla producto x mes con todas las combinaciones (meses sin venta = 0)."""
    meses = pd.date_range(df["mes"].min(), df["mes"].max(), freq="MS")
    idx = pd.MultiIndex.from_product([df["producto"].unique(), meses], names=["producto", "mes"])
    m = df.groupby(["producto", "mes"])["cantidad"].sum().reindex(idx, fill_value=0)
    return m.reset_index()


# ---------------------------------------------------------------- ML 1: clasificación
@st.cache_data
def clasificar_rotacion(df: pd.DataFrame, n_clusters: int = 3):
    dias_totales = df["fecha"].nunique()
    mensual = ventas_mensuales(df)
    feats = df.groupby("producto").agg(
        categoria=("categoria", "first"),
        unidades=("cantidad", "sum"),
        ingresos=("total", "sum"),
        dias_con_venta=("fecha", "nunique"),
    )
    feats["unidades_mes"] = mensual.groupby("producto")["cantidad"].mean()
    feats["frecuencia_%"] = 100 * feats["dias_con_venta"] / dias_totales
    cv = mensual.groupby("producto")["cantidad"].agg(lambda s: s.std() / s.mean() if s.mean() else 0)
    feats["variabilidad"] = cv

    # La rotación se define por cuánto y con qué frecuencia se vende (ingresos y variabilidad se muestran
    # como contexto: incluirlos mezcla "rotación" con "valor" y separa sólo los atípicos)
    X = feats[["unidades_mes", "frecuencia_%"]].copy()
    X["unidades_mes"] = np.log1p(X["unidades_mes"])  # reduce el sesgo de los productos estrella
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    feats["cluster"] = km.fit_predict(Xs)
    sil = silhouette_score(Xs, feats["cluster"]) if len(feats) > n_clusters else np.nan

    # Ordenar clusters por unidades/mes para ponerles nombre
    orden = feats.groupby("cluster")["unidades_mes"].mean().sort_values(ascending=False).index
    nombres = ["Alta", "Media", "Baja"] if n_clusters == 3 else [f"Nivel {i+1}" for i in range(n_clusters)]
    feats["rotacion"] = feats["cluster"].map(dict(zip(orden, nombres)))
    return feats.drop(columns="cluster").reset_index(), sil


# ---------------------------------------------------------------- ML 2: predicción
def _features(serie: pd.DataFrame) -> pd.DataFrame:
    s = serie.sort_values("mes").copy()
    g = s.groupby("producto")["cantidad"]
    for k in (1, 2, 3):
        s[f"lag{k}"] = g.shift(k)
    s["media3"] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    s["tendencia"] = s["lag1"] - s["lag3"]
    s["mes_sin"] = np.sin(2 * np.pi * s["mes"].dt.month / 12)
    s["mes_cos"] = np.cos(2 * np.pi * s["mes"].dt.month / 12)
    s["nivel_producto"] = g.transform(lambda x: x.shift(1).expanding().mean())
    return s


FEATS = ["lag1", "lag2", "lag3", "media3", "tendencia", "mes_sin", "mes_cos", "nivel_producto"]


@st.cache_data
def predecir_proximo_mes(df: pd.DataFrame):
    mensual = ventas_mensuales(df)
    proximo = mensual["mes"].max() + pd.offsets.MonthBegin(1)

    # fila "futura" vacía por producto para generar sus features
    futuro = pd.DataFrame({"producto": mensual["producto"].unique(), "mes": proximo, "cantidad": np.nan})
    todo = _features(pd.concat([mensual, futuro], ignore_index=True))
    entren = todo[todo["cantidad"].notna()].dropna(subset=FEATS)
    pred_rows = todo[todo["mes"] == proximo]

    # Validación: entrenar sin el último mes y medir error sobre él
    ultimo = entren["mes"].max()
    tr, te = entren[entren["mes"] < ultimo], entren[entren["mes"] == ultimo]
    modelo = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)
    modelo.fit(tr[FEATS], tr["cantidad"])
    mae = mean_absolute_error(te["cantidad"], modelo.predict(te[FEATS]))
    wape = 100 * np.abs(te["cantidad"] - modelo.predict(te[FEATS])).sum() / te["cantidad"].sum()

    # Modelo final con todos los datos
    modelo.fit(entren[FEATS], entren["cantidad"])
    pred = pred_rows[["producto"]].copy()
    pred["prediccion"] = np.clip(modelo.predict(pred_rows[FEATS]), 0, None)

    # error típico por producto (desv. estándar de los últimos 6 meses) para stock de seguridad
    pred["desv"] = pred["producto"].map(
        mensual.sort_values("mes").groupby("producto")["cantidad"].apply(lambda s: s.tail(6).std())
    )
    importancias = pd.Series(modelo.feature_importances_, index=FEATS).sort_values()
    return pred, proximo, mae, wape, importancias, mensual


# ================================================================= UI
st.title("📦 Análisis de ventas con Machine Learning")
st.caption("Sube tu CSV de ventas o usa los datos de ejemplo. "
           "Columnas: fecha, producto, cantidad, precio_unitario (categoria es opcional).")

with st.sidebar:
    st.header("⚙️ Configuración")
    archivo = st.file_uploader("CSV de ventas", type="csv")
    st.divider()
    nivel_servicio = st.select_slider(
        "Nivel de servicio (stock de seguridad)",
        options=["Bajo (80%)", "Medio (90%)", "Alto (95%)", "Muy alto (99%)"], value="Alto (95%)",
    )
    z = {"Bajo (80%)": 0.84, "Medio (90%)": 1.28, "Alto (95%)": 1.65, "Muy alto (99%)": 2.33}[nivel_servicio]

try:
    df = cargar_datos(archivo if archivo else "ventas.csv")
except FileNotFoundError:
    st.error("No se encontró ventas.csv. Ejecuta `python generar_datos.py` o sube un CSV.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

if archivo is None:
    st.info("Usando datos de ejemplo (ventas.csv).", icon="ℹ️")

cats = sorted(df["categoria"].unique())
with st.sidebar:
    cats_sel = st.multiselect("Categorías", cats, default=cats)
df_f = df[df["categoria"].isin(cats_sel)]
if df_f.empty:
    st.warning("Selecciona al menos una categoría.")
    st.stop()

# ---------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ingresos totales", f"S/ {df_f['total'].sum():,.0f}")
c2.metric("Unidades vendidas", f"{df_f['cantidad'].sum():,}")
c3.metric("Productos", df_f["producto"].nunique())
c4.metric("Periodo", f"{df_f['fecha'].min():%m/%Y} – {df_f['fecha'].max():%m/%Y}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Ventas", "🔄 Rotación (ML)", "🔮 Predicción y compras", "🗂️ Datos"])

# ---------------- Tab 1: ventas
with tab1:
    mens = df_f.groupby(["mes", "categoria"])["total"].sum().reset_index()
    fig = px.area(mens, x="mes", y="total", color="categoria", title="Ingresos mensuales por categoría",
                  labels={"mes": "", "total": "Ingresos (S/)", "categoria": "Categoría"})
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    top = df_f.groupby("producto")["total"].sum().sort_values().tail(10).reset_index()
    a.plotly_chart(px.bar(top, x="total", y="producto", orientation="h", title="Top 10 productos por ingresos",
                          labels={"total": "Ingresos (S/)", "producto": ""}), width="stretch")
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    dsem = df_f.groupby(df_f["fecha"].dt.dayofweek)["total"].sum().reindex(range(7), fill_value=0)
    b.plotly_chart(px.bar(x=dias, y=dsem.values, title="Ingresos por día de la semana",
                          labels={"x": "", "y": "Ingresos (S/)"}), width="stretch")

# ---------------- Tab 2: clasificación
with tab2:
    if df_f["producto"].nunique() < 4:
        st.warning("Se necesitan al menos 4 productos para clasificar.")
    else:
        clas, sil = clasificar_rotacion(df_f)
        st.markdown(
            "Se agrupan los productos con **K-Means** según sus unidades vendidas por mes y su "
            "frecuencia de venta (% de días con al menos una venta). "
            f"Calidad del agrupamiento (silhouette): **{sil:.2f}** (de -1 a 1; >0.5 es bueno)."
        )
        cols = st.columns(3)
        for col, nivel in zip(cols, ["Alta", "Media", "Baja"]):
            sub = clas[clas["rotacion"] == nivel]
            col.metric(f"Rotación {nivel}", f"{len(sub)} productos",
                       f"{100*sub['ingresos'].sum()/clas['ingresos'].sum():.0f}% de ingresos", delta_color="off")

        fig = px.scatter(clas, x="frecuencia_%", y="unidades_mes", size="ingresos", color="rotacion",
                         hover_name="producto", log_y=True, color_discrete_map=COLORES_ROTACION,
                         category_orders={"rotacion": ["Alta", "Media", "Baja"]},
                         title="Mapa de rotación de productos",
                         labels={"frecuencia_%": "Frecuencia de venta (% de días)",
                                 "unidades_mes": "Unidades por mes (escala log)", "rotacion": "Rotación"})
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            clas.sort_values("unidades_mes", ascending=False)[
                ["producto", "categoria", "rotacion", "unidades_mes", "frecuencia_%", "ingresos", "variabilidad"]],
            hide_index=True, width="stretch",
            column_config={
                "unidades_mes": st.column_config.NumberColumn("Unid./mes", format="%.0f"),
                "frecuencia_%": st.column_config.ProgressColumn("Frecuencia", format="%.0f%%", min_value=0, max_value=100),
                "ingresos": st.column_config.NumberColumn("Ingresos", format="S/ %.0f"),
                "variabilidad": st.column_config.NumberColumn("Variabilidad (CV)", format="%.2f"),
            },
        )
        with st.expander("💡 ¿Qué hacer con cada grupo?"):
            st.markdown(
                "- **Alta rotación:** nunca quedarse sin stock; reponer con frecuencia, negociar precio por volumen.\n"
                "- **Media rotación:** reposición regular según la predicción.\n"
                "- **Baja rotación:** comprar poco, evaluar promociones o retirarlos; ojo con productos "
                "estacionales (alta variabilidad), que pueden parecer de baja rotación fuera de temporada."
            )

# ---------------- Tab 3: predicción
with tab3:
    pred, proximo, mae, wape, imp, mensual = predecir_proximo_mes(df_f)
    pred["stock_seguridad"] = z * pred["desv"].fillna(0)
    pred["compra_sugerida"] = np.ceil(pred["prediccion"] + pred["stock_seguridad"]).astype(int)
    precios = df_f.groupby("producto")["precio_unitario"].mean()
    pred["valor_estimado"] = pred["prediccion"] * pred["producto"].map(precios)

    st.markdown(
        f"Modelo **Random Forest** entrenado con las ventas mensuales de todos los productos "
        f"(últimos 3 meses, tendencia y mes del año). Predicción para **{MESES[proximo.month-1]} {proximo.year}**."
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Unidades previstas", f"{pred['prediccion'].sum():,.0f}")
    k2.metric("Ingresos previstos", f"S/ {pred['valor_estimado'].sum():,.0f}")
    k3.metric("Error del modelo (último mes)", f"{wape:.0f}%", f"MAE {mae:.0f} unid./producto", delta_color="off")

    tabla = pred.sort_values("compra_sugerida", ascending=False)[
        ["producto", "prediccion", "stock_seguridad", "compra_sugerida", "valor_estimado"]]
    st.dataframe(tabla, hide_index=True, width="stretch", column_config={
        "prediccion": st.column_config.NumberColumn("Predicción (unid.)", format="%.0f"),
        "stock_seguridad": st.column_config.NumberColumn(f"Stock seguridad ({nivel_servicio})", format="%.0f"),
        "compra_sugerida": st.column_config.NumberColumn("🛒 Comprar", format="%d"),
        "valor_estimado": st.column_config.NumberColumn("Venta estimada", format="S/ %.0f"),
    })
    st.download_button("⬇️ Descargar plan de compras (CSV)", tabla.round(1).to_csv(index=False).encode("utf-8"),
                       file_name=f"plan_compras_{proximo:%Y_%m}.csv", mime="text/csv")

    st.subheader("Detalle por producto")
    prod = st.selectbox("Producto", sorted(pred["producto"]))
    hist = mensual[mensual["producto"] == prod]
    p = pred[pred["producto"] == prod].iloc[0]
    fig = go.Figure()
    fig.add_scatter(x=hist["mes"], y=hist["cantidad"], mode="lines+markers", name="Histórico")
    fig.add_scatter(x=[hist["mes"].iloc[-1], proximo], y=[hist["cantidad"].iloc[-1], p["prediccion"]],
                    mode="lines+markers", name="Predicción", line=dict(dash="dash", color="#e76f51"))
    fig.add_bar(x=[proximo], y=[p["compra_sugerida"]], name="Compra sugerida", marker_color="rgba(42,157,143,0.35)")
    fig.update_layout(title=f"{prod}: unidades por mes", yaxis_title="Unidades", xaxis_title="")
    st.plotly_chart(fig, width="stretch")

    with st.expander("🔍 ¿En qué se fija el modelo?"):
        st.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h",
                               labels={"x": "Importancia", "y": ""}), width="stretch")
        st.caption("Con solo 12 meses de historia el modelo no puede aprender la estacionalidad anual "
                   "(p. ej. el pico navideño del panetón). Con 2+ años de datos mejora mucho.")

# ---------------- Tab 4: datos
with tab4:
    st.dataframe(df_f.drop(columns="mes").sort_values("fecha", ascending=False), hide_index=True,
                 width="stretch")
