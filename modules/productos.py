"""
Sección 2: PRODUCTOS (catálogo, ficha integrada de cada producto e importación de datos).
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.inventario import obtener_tabla
from modules.rotacion import ICONOS as ICONOS_ROTACION
from modules.rotacion import ejecutar_modelo
from utils import importacion
from utils.autenticacion import puede_editar
from utils.abastecimiento import ACCIONES, recomendar
from utils.componentes import (AZUL, GRIS_GRILLA, etiqueta_estado, estilo_grafico, formato_soles, tarjeta)
from utils.ml_demanda import HORIZONTES


def mostrar(datos, filtros):
    st.title("Productos")
    catalogo, ficha, importar = st.tabs(["📋 Catálogo", "🔎 Ficha del producto", "📥 Importar datos"])

    integrado = combinar_resultados(datos)
    with catalogo:
        mostrar_catalogo(datos, filtros, integrado)
    with ficha:
        mostrar_ficha(datos, filtros, integrado)
    with importar:
        mostrar_importacion(datos)


def combinar_resultados(datos):
    """Une en una sola tabla los resultados de rotación, predicción, stock y abastecimiento."""
    productos = datos["productos"].copy()
    productos["producto"] = productos["nombre"] + " " + productos["talla"] + " " + productos["color"]
    productos["margen_pct"] = (productos["precio_venta"] - productos["precio_compra"]) / productos["precio_venta"] * 100
    productos["valor_costo"] = productos["stock_actual"] * productos["precio_compra"]
    resultado = {"tabla": productos, "res": None, "stock": None, "plan": None}
    ventas = datos["ventas"]
    try:
        rot, _, _ = ejecutar_modelo(ventas, datos["productos"], ventas["fecha"].min().date(), ventas["fecha"].max().date())
        productos = productos.merge(rot[["codigo", "rotacion"]], on="codigo", how="left")
    except ValueError:
        productos["rotacion"] = None
    try:
        stock, res = obtener_tabla(datos, st.session_state.get("nivel_servicio", 95),
                                   st.session_state.get("dias_a_cubrir", 30), st.session_state.get("dias_exceso", 90))
        plan = recomendar(stock, datos["productos"], res, st.session_state.get("dias_a_cubrir", 30),
                          st.session_state.get("dias_exceso", 90))
        productos = productos.merge(stock[["codigo", "estado"]], on="codigo", how="left")
        productos = productos.merge(res["predicciones"][["codigo", "pred_30"]], on="codigo", how="left")
        resultado.update({"res": res, "stock": stock, "plan": plan})
    except ValueError:
        productos["estado"], productos["pred_30"] = None, None
    resultado["tabla"] = productos
    return resultado


def mostrar_catalogo(datos, filtros, integrado):
    tabla = integrado["tabla"]
    if filtros["categorias"]:
        tabla = tabla[tabla["categoria"].isin(filtros["categorias"])]
    if filtros["codigos"]:
        tabla = tabla[tabla["codigo"].isin(filtros["codigos"])]

    c1, c2, c3, c4 = st.columns(4)
    tarjeta(c1, "Productos", f"{len(tabla)}", f"{tabla['categoria'].nunique()} categorías")
    tarjeta(c2, "Unidades en stock", f"{int(tabla['stock_actual'].sum()):,}")
    tarjeta(c3, "Valor del inventario", formato_soles(tabla["valor_costo"].sum()),
            f"A precio de venta: {formato_soles((tabla['stock_actual'] * tabla['precio_venta']).sum())}")
    tarjeta(c4, "Margen promedio", f"{tabla['margen_pct'].mean():.1f} %", "(precio venta − compra) ÷ precio venta")

    vista = tabla.copy()
    vista["Rotación"] = vista["rotacion"].map(lambda r: f"{ICONOS_ROTACION[r]} {r}" if isinstance(r, str) else "—")
    vista["Estado de stock"] = vista["estado"].map(lambda e: etiqueta_estado(e) if isinstance(e, str) else "—")
    st.dataframe(vista[["codigo", "nombre", "categoria", "talla", "color", "precio_compra", "precio_venta",
                        "margen_pct", "stock_actual", "stock_minimo", "tiempo_reposicion_dias",
                        "Rotación", "pred_30", "Estado de stock"]],
                 hide_index=True, width="stretch", column_config={
                     "codigo": "Código", "nombre": "Producto", "categoria": "Categoría", "talla": "Talla",
                     "color": "Color",
                     "precio_compra": st.column_config.NumberColumn("P. compra", format="S/ %.2f"),
                     "precio_venta": st.column_config.NumberColumn("P. venta", format="S/ %.2f"),
                     "margen_pct": st.column_config.NumberColumn("Margen", format="%.0f %%"),
                     "stock_actual": st.column_config.NumberColumn("Stock", format="%d"),
                     "stock_minimo": st.column_config.NumberColumn("Stock mín.", format="%d"),
                     "tiempo_reposicion_dias": st.column_config.NumberColumn("Reposición (días)", format="%d"),
                     "pred_30": st.column_config.NumberColumn("Demanda 30 días (modelo)", format="%.0f")})

    st.subheader("Valor del inventario por categoría")
    por_cat = (tabla.groupby("categoria").agg(valor=("valor_costo", "sum"), unidades=("stock_actual", "sum"))
               .sort_values("valor").reset_index())
    fig = px.bar(por_cat, x="valor", y="categoria", orientation="h", custom_data=["unidades"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>S/ %{x:,.0f} · %{customdata[0]} unidades<extra></extra>")
    fig = estilo_grafico(fig, alto=380)
    fig.update_xaxes(tickprefix="S/ ", tickformat=",.0f", showgrid=True, gridcolor=GRIS_GRILLA)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")


def mostrar_ficha(datos, filtros, integrado):
    tabla = integrado["tabla"]
    if filtros["categorias"]:
        tabla = tabla[tabla["categoria"].isin(filtros["categorias"])]
    if tabla.empty:
        st.info("No hay productos con los filtros elegidos.")
        return
    codigo = st.selectbox("Producto", tabla["codigo"],
                          format_func=dict(zip(tabla["codigo"], tabla["codigo"] + " · " + tabla["producto"])).get)
    p = tabla.set_index("codigo").loc[codigo]
    st.markdown(f"### {p['producto']}")
    st.caption(f"{p['categoria']} · Proveedor: {p.get('proveedor', '—')} · Reposición: "
               f"{int(p['tiempo_reposicion_dias'])} días")

    c1, c2, c3, c4 = st.columns(4)
    tarjeta(c1, "Precios", f"S/ {p['precio_venta']:.2f}", f"Compra S/ {p['precio_compra']:.2f} · margen {p['margen_pct']:.0f} %")
    tarjeta(c2, "Stock actual", f"{int(p['stock_actual'])} unid.", f"Valor a costo: {formato_soles(p['valor_costo'])}")
    rot = p.get("rotacion")
    tarjeta(c3, "Rotación (K-Means)", f"{ICONOS_ROTACION[rot]} {rot}" if isinstance(rot, str) else "—",
            "Clasificación con todo el historial")
    estado = p.get("estado")
    tarjeta(c4, "Estado de stock", etiqueta_estado(estado) if isinstance(estado, str) else "—")

    if integrado["res"] is not None:
        pred = integrado["res"]["predicciones"].set_index("codigo").loc[codigo]
        st.markdown("**Demanda estimada por el modelo**")
        cols = st.columns(len(HORIZONTES))
        for col, h in zip(cols, HORIZONTES):
            tarjeta(col, f"{h} días", f"{pred[f'pred_{h}']:.0f} unid.",
                    f"Rango {pred[f'min_{h}']:.0f} – {pred[f'max_{h}']:.0f}")
        rec = integrado["plan"].set_index("codigo").loc[codigo]
        s = integrado["stock"].set_index("codigo").loc[codigo]
        texto = f"**Recomendación:** {ACCIONES[rec['accion']]} {rec['accion']}"
        if pd.notna(rec["fecha_pedido"]):
            texto += (f" — pedir **{rec['cantidad']} unidades** el **{rec['fecha_pedido']:%d/%m/%Y}** "
                      f"(llegarían aprox. el {rec['fecha_llegada']:%d/%m/%Y}, costo {formato_soles(rec['costo'])})")
        st.info(texto + f". {rec['motivo']}")
        st.caption(f"Se agota aprox. en {min(s['dias_hasta_agotar'], 999):.0f} días · punto de pedido "
                   f"{s['punto_pedido']} · stock de seguridad {s['stock_seguridad']}.")

    ventas = datos["ventas"]
    v = ventas[ventas["codigo"] == codigo]
    semanal = v.set_index("fecha")["cantidad"].resample("W").sum()
    st.markdown("**Ventas por semana (unidades)**")
    if semanal.empty:
        st.info("Este producto no tiene ventas registradas.")
    else:
        fig = px.bar(semanal.reset_index(), x="fecha", y="cantidad")
        fig.update_traces(hovertemplate="Semana del %{x|%d/%m/%Y}<br>%{y} unidades<extra></extra>", marker_color=AZUL)
        st.plotly_chart(estilo_grafico(fig, alto=280), width="stretch")


def mostrar_importacion(datos):
    if not puede_editar():
        st.info("🔒 Tu rol (**empleado**) permite ver los datos, pero no importarlos ni reemplazarlos. "
                "Pídeselo al dueño de la tienda.")
        return
    destino = ("la **base de datos** (solo los de tu tienda)" if importacion.base_datos.configurada()
               else "la carpeta **data** de tu computadora")
    st.markdown(f"Sube tus propios archivos **CSV o Excel** para reemplazar los datos actuales en {destino}. "
                "El sistema los **valida antes de guardarlos** y guarda una copia de respaldo de los datos "
                "anteriores en `data/respaldos/`.")
    tipo = st.radio("¿Qué archivo vas a importar?", list(importacion.DESCRIPCION), horizontal=True,
                    format_func=str.capitalize)
    st.caption(importacion.DESCRIPCION[tipo])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Columnas obligatorias:** " + ", ".join(f"`{c}`" for c in importacion.COLUMNAS[tipo]))
        st.markdown("**Opcionales:** " + ", ".join(f"`{c}`" for c in importacion.OPCIONALES[tipo]))
        st.caption("Fechas aceptadas: 2026-11-18 o 18/11/2026. Los números sin símbolo S/ ni separador de miles.")
    with col_b:
        st.download_button(f"⬇️ Descargar plantilla de {tipo} (CSV)",
                           importacion.plantilla(tipo).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"plantilla_{tipo}.csv", mime="text/csv")

    archivo = st.file_uploader("Sube el archivo", type=["csv", "xlsx"], key=f"subir_{tipo}")
    if archivo is not None:
        tabla, errores, advertencias, resumen = importacion.validar_archivo(archivo, tipo, datos["productos"])
        if resumen:
            st.write(" · ".join(f"**{k}:** {v}" for k, v in resumen.items()))
        for e in errores:
            st.error(f"❌ {e}")
        for a in advertencias:
            st.warning(f"⚠️ {a}")
        if tabla is not None:
            st.dataframe(tabla.head(20), hide_index=True, width="stretch")
        if not errores:
            st.success("✅ El archivo pasó todas las validaciones.")
            if st.button(f"💾 Guardar como nuevo archivo de {tipo} y actualizar el sistema", type="primary"):
                with st.spinner("Guardando..."):
                    destino, respaldo = importacion.guardar(tabla, tipo)
                limpiar_y_recargar(f"Guardado en {destino}." +
                                   (f" Respaldo del anterior: `respaldos/{respaldo.name}`." if respaldo else ""))
        else:
            st.info("Corrige los errores en tu archivo y vuelve a subirlo. Nada se guardó.")

    if "mensaje_importacion" in st.session_state:
        st.success(st.session_state.pop("mensaje_importacion"))

    st.divider()
    with st.expander("♻️ Volver a los datos ficticios de prueba"):
        st.write("Genera los datos ficticios de prueba (productos, ventas y compras) y reemplaza los "
                 "actuales. Antes se guarda una copia de respaldo en `data/respaldos/`.")
        if st.button("Cargar datos ficticios de prueba"):
            with st.spinner("Generando y guardando los datos de prueba..."):
                if not importacion.base_datos.configurada():
                    for tipo_archivo in importacion.DESCRIPCION:   # respaldo de los CSV actuales
                        ruta = importacion.CARPETA_DATOS / f"{tipo_archivo}.csv"
                        if ruta.exists():
                            importacion.guardar(pd.read_csv(ruta, encoding="utf-8-sig"), tipo_archivo,
                                                ficticios=True)
                mensaje = importacion.cargar_datos_de_prueba()
            limpiar_y_recargar(mensaje)


def limpiar_y_recargar(mensaje):
    """Borra la memoria (caché) para que todas las secciones lean los datos nuevos y el modelo se reentrene."""
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["mensaje_importacion"] = mensaje + " Los modelos se reentrenarán al abrir sus secciones."
    st.rerun()
