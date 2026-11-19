"""
Sección 4: ROTACIÓN DE INVENTARIO (clasificación con K-Means).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.componentes import fuente, tarjeta
from utils import ml_rotacion as ml

# Un color fijo por clase (siempre el mismo, aunque se filtren datos)
COLORES = {"Alta rotación": "#2a78d6", "Rotación media": "#eb6834", "Baja rotación": "#1baf7a"}
ICONOS = {"Alta rotación": "🔵", "Rotación media": "🟠", "Baja rotación": "🟢"}
ARCHIVO_REFERENCIA = Path(__file__).resolve().parent.parent / "data" / "simulacion_referencia.csv"


@st.cache_data(show_spinner="Entrenando el modelo de rotación...")
def ejecutar_modelo(ventas, productos, fecha_inicio, fecha_fin):
    variables = ml.calcular_variables(ventas, productos, fecha_inicio, fecha_fin)
    return ml.clasificar(variables)


def mostrar(datos, filtros):
    st.title("Rotación de inventario")
    st.caption(f"Periodo analizado: {filtros['fecha_inicio']:%d/%m/%Y} al {filtros['fecha_fin']:%d/%m/%Y}  ·  "
               f"Clasificación calculada por el modelo K-Means  ·  {fuente(datos)}")

    try:
        tabla, calidad, resumen = ejecutar_modelo(
            datos["ventas"], datos["productos"], filtros["fecha_inicio"], filtros["fecha_fin"])
    except ValueError as e:
        st.warning(f"No se pudo clasificar: {e} Amplía el rango de fechas.")
        return

    if (pd.Timestamp(filtros["fecha_fin"]) - pd.Timestamp(filtros["fecha_inicio"])).days < 60:
        st.warning("El periodo elegido tiene menos de 60 días. Con tan poco historial la "
                   "clasificación es poco confiable; se recomienda usar 3 meses o más.")

    # El modelo siempre se entrena con TODOS los productos; los filtros solo
    # cambian lo que se muestra (si no, los grupos cambiarían al filtrar).
    visibles = tabla
    if filtros["categorias"]:
        visibles = visibles[visibles["categoria"].isin(filtros["categorias"])]
    if filtros["codigos"]:
        visibles = visibles[visibles["codigo"].isin(filtros["codigos"])]
    if filtros["categorias"] or filtros["codigos"]:
        st.info("El modelo se entrenó con todos los productos; aquí solo se muestran los filtrados.")

    # ---------------- Tarjetas ----------------
    c1, c2, c3, c4 = st.columns(4)
    for col, clase in zip((c1, c2, c3), ml.CLASES):
        fila = resumen[resumen["rotacion"] == clase]
        n = int(fila["productos"].iloc[0]) if not fila.empty else 0
        part = fila["participacion_ventas"].iloc[0] if not fila.empty else 0
        tarjeta(col, f"{ICONOS[clase]} {clase}", f"{n} productos",
                f"{part:.0f} % de las unidades vendidas")
    tarjeta(c4, "Calidad de los grupos (silhouette)", f"{calidad:.2f}",
            f"{ml.interpretar_silhouette(calidad).capitalize()}. Escala de -1 a 1: "
            "sobre 0.25 es aceptable; sobre 0.5, bueno.")

    # ---------------- Gráfico de grupos ----------------
    st.subheader("Mapa de productos")
    st.caption("Cada punto es un producto. Más a la derecha = vende más unidades por día; "
               "más arriba = se vende en más días. Pasa el cursor para ver el detalle.")
    fig = px.scatter(
        visibles, x="unidades_por_dia", y="frecuencia_venta", color="rotacion",
        color_discrete_map=COLORES, category_orders={"rotacion": ml.CLASES}, log_x=True,
        custom_data=["producto", "codigo", "dias_entre_ventas", "cobertura_dias"],
        labels={"unidades_por_dia": "Unidades por día (escala logarítmica)",
                "frecuencia_venta": "% de días con venta", "rotacion": "Clase"})
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color="white")),
                      hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                    "Unidades por día: %{x:.2f}<br>Días con venta: %{y:.0f} %<br>"
                                    "Días entre ventas: %{customdata[2]:.1f}<br>"
                                    "Cobertura: %{customdata[3]:.0f} días<extra></extra>")
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(color="#52514e", size=13), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", y=1.08, x=0, title=None))
    fig.update_xaxes(gridcolor="#ececea", tickvals=[0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
                     ticktext=["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    fig.update_yaxes(gridcolor="#ececea", ticksuffix=" %", rangemode="tozero")
    st.plotly_chart(fig, width="stretch")

    # ---------------- Perfil de cada grupo ----------------
    st.subheader("Perfil de cada grupo")
    st.caption("Promedios de las variables en cada grupo: así se ve qué distingue a cada clase.")
    perfil = resumen.rename(columns={
        "rotacion": "Clase", "productos": "Productos",
        "unidades_por_dia": "Unid./día (prom.)", "frecuencia_venta": "% días con venta",
        "dias_entre_ventas": "Días entre ventas", "cobertura_dias": "Cobertura (mediana, días)",
        "unidades_vendidas": "Unidades vendidas", "participacion_ventas": "% de unidades"})
    perfil["Clase"] = perfil["Clase"].map(lambda c: f"{ICONOS[c]} {c}")
    st.dataframe(perfil, hide_index=True, width="stretch", column_config={
        "Productos": st.column_config.NumberColumn(format="%d"),
        "Unid./día (prom.)": st.column_config.NumberColumn(format="%.2f"),
        "% días con venta": st.column_config.NumberColumn(format="%.0f %%"),
        "Días entre ventas": st.column_config.NumberColumn(format="%.1f"),
        "Cobertura (mediana, días)": st.column_config.NumberColumn(format="%.0f"),
        "Unidades vendidas": st.column_config.NumberColumn(format="%d"),
        "% de unidades": st.column_config.NumberColumn(format="%.0f %%")})

    # ---------------- Tabla detallada ----------------
    st.subheader("Clasificación por producto")
    elegidas = st.multiselect("Mostrar clases", ml.CLASES, default=ml.CLASES)
    detalle = visibles[visibles["rotacion"].isin(elegidas)].sort_values("unidades_por_dia", ascending=False).copy()
    detalle["Clase"] = detalle["rotacion"].map(lambda c: f"{ICONOS[c]} {c}")
    detalle["Nota"] = detalle["historial_corto"].map(
        lambda corto: "⚠️ Poco historial" if corto else "")
    st.dataframe(
        detalle[["Clase", "codigo", "producto", "categoria", "unidades_vendidas", "unidades_por_dia",
                 "frecuencia_venta", "dias_entre_ventas", "dias_desde_ultima_venta",
                 "stock_actual", "cobertura_dias", "Nota"]],
        hide_index=True, width="stretch", column_config={
            "codigo": "Código", "producto": "Producto", "categoria": "Categoría",
            "unidades_vendidas": st.column_config.NumberColumn("Unid. vendidas", format="%d"),
            "unidades_por_dia": st.column_config.NumberColumn("Unid./día", format="%.2f"),
            "frecuencia_venta": st.column_config.NumberColumn("% días con venta", format="%.0f %%"),
            "dias_entre_ventas": st.column_config.NumberColumn("Días entre ventas", format="%.1f"),
            "dias_desde_ultima_venta": st.column_config.NumberColumn("Días desde últ. venta", format="%d"),
            "stock_actual": st.column_config.NumberColumn("Stock", format="%d"),
            "cobertura_dias": st.column_config.NumberColumn(
                "Cobertura (días)", format="%.0f",
                help="Para cuántos días alcanza el stock actual al ritmo de venta promedio. "
                     "Se limita a 365."),
        })
    if detalle["historial_corto"].any():
        st.caption(f"⚠️ Poco historial: el producto tiene menos de {ml.HISTORIAL_CORTO} días de "
                   "ventas en el periodo; su clase puede cambiar cuando acumule más datos.")

    # ---------------- Explicación del algoritmo ----------------
    with st.expander("ℹ️ ¿Cómo se determina la rotación? (algoritmo, motivos y limitaciones)"):
        st.markdown(f"""
**Algoritmo: K-Means (agrupamiento no supervisado)**

1. **Variables.** Para cada producto se calculan, con las ventas del periodo:
   unidades por día, % de días con venta, días promedio entre ventas, días desde la
   última venta y días de cobertura (stock actual ÷ unidades por día).
   Si un producto empezó a venderse después del inicio del periodo, solo se cuentan
   los días desde su fecha de alta.
2. **Preparación.** Se aplica logaritmo a las variables muy dispares y luego se
   *estandarizan* (promedio 0, desviación 1) para que todas pesen lo mismo.
3. **Agrupamiento.** K-Means busca **3 grupos** de productos con comportamiento parecido,
   minimizando la distancia de cada producto al centro de su grupo. Se ejecuta 20 veces
   con distintos puntos de partida y se queda con la mejor solución.
4. **Nombre de los grupos.** El grupo con mayor promedio de unidades por día se llama
   *Alta rotación*; el menor, *Baja rotación*; el restante, *Rotación media*.
   No se usan umbrales fijos inventados (como "más de 1 unidad al día").

**¿Por qué K-Means?** No necesita productos etiquetados de antemano (la tienda no tiene
esa información), es rápido, fácil de explicar y combina varias variables a la vez en
lugar de mirar solo las unidades vendidas.

**Calidad.** El coeficiente *silhouette* de este resultado es **{calidad:.2f}**
({ml.interpretar_silhouette(calidad)}).

**Limitaciones**
- Siempre forma 3 grupos, aunque los datos se separen mejor en otro número.
- Los productos que quedan en la frontera entre dos grupos pueden cambiar de clase
  si se modifica el periodo analizado.
- La **cobertura** influye bastante: un producto con mucho stock acumulado tiende a
  caer en *Baja rotación*, lo que es útil para detectar inventario inmovilizado.
- Con poco historial (productos nuevos o periodos cortos) los resultados son menos confiables.
- Con muy pocos productos (menos de 6) no se puede aplicar.
- Promedia todo el periodo: un producto de temporada (ej. casacas) se evalúa por su
  promedio anual, no por su mejor mes. La estacionalidad se analizará en la Etapa 8.
""")

    # ---------------- Validación con la simulación ----------------
    if datos["ficticios"] and ARCHIVO_REFERENCIA.exists():
        with st.expander("🧪 Validación: ¿el modelo encontró los perfiles que se simularon?"):
            mostrar_validacion(tabla)


def mostrar_validacion(tabla):
    """
    Compara la clasificación del modelo con la demanda que se usó para SIMULAR
    los datos. El modelo nunca ve este archivo: es solo para comprobar.
    """
    ref = pd.read_csv(ARCHIVO_REFERENCIA, encoding="utf-8-sig")
    comp = tabla.merge(ref[["codigo", "demanda_base_simulada"]], on="codigo")
    # En el generador: alta = 1.5 o más unidades/día de catálogo, baja = menos de 0.5,
    # ambos multiplicados por la escala 0.6 -> 0.9 y 0.3 unidades/día.
    comp["clase_simulada"] = pd.cut(comp["demanda_base_simulada"], [-1, 0.3, 0.9, 999], right=False,
                                    labels=["Baja rotación", "Rotación media", "Alta rotación"]).astype(str)
    coincide = comp["clase_simulada"] == comp["rotacion"]

    st.markdown(f"""
Este archivo (`simulacion_referencia.csv`) contiene la demanda **con la que se inventaron**
los datos. El modelo **no lo usó** para entrenar; aquí solo sirve para verificar.

Se considera *simulada alta* una demanda base de 0.9 unidades/día o más, y *simulada baja*
menos de 0.3.

**Coincidencias: {coincide.sum()} de {len(comp)} productos ({coincide.mean() * 100:.0f} %).**
""")
    diferencias = comp[~coincide]
    if not diferencias.empty:
        st.markdown("Productos donde el modelo difiere de la simulación (suelen ser casos de frontera):")
        st.dataframe(diferencias[["codigo", "producto", "demanda_base_simulada", "clase_simulada",
                                  "rotacion", "unidades_por_dia", "cobertura_dias"]],
                     hide_index=True, width="stretch", column_config={
                         "codigo": "Código", "producto": "Producto",
                         "demanda_base_simulada": st.column_config.NumberColumn("Demanda simulada", format="%.2f"),
                         "clase_simulada": "Clase según simulación", "rotacion": "Clase según modelo",
                         "unidades_por_dia": st.column_config.NumberColumn("Unid./día real", format="%.2f"),
                         "cobertura_dias": st.column_config.NumberColumn("Cobertura (días)", format="%.0f")})
