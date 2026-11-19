"""
PREDICTA RETAIL - Sistema inteligente de gestión de inventarios.

Archivo principal: arma el menú lateral, carga los datos, aplica los filtros
y muestra la sección elegida.

Para ejecutarlo:  streamlit run app.py
"""

import streamlit as st

from modules import (abastecimiento, dashboard, inventario, prediccion, productos, reportes, rotacion,
                     ventas)
from utils.autenticacion import controlar_acceso, mostrar_usuario
from utils.componentes import evitar_traduccion, fuente, origen
from utils.data_loader import cargar_datos

st.set_page_config(page_title="Predicta Retail", page_icon="📊", layout="wide")

# Pequeños ajustes visuales
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stSidebar"] h1 { font-size: 1.6rem; color: #2a78d6; }
</style>
""", unsafe_allow_html=True)

# Cada opción del menú y el módulo que la muestra
SECCIONES = {
    "📊 Panel de control": dashboard,
    "👕 Productos": productos,
    "🧾 Ventas y temporadas": ventas,
    "🔄 Rotación de inventario": rotacion,
    "📈 Predicción de demanda": prediccion,
    "📦 Control de stock": inventario,
    "🛒 Abastecimiento inteligente": abastecimiento,
    "📑 Reportes": reportes,
}

# ------------------------------------------------------------------
# Inicio de sesión: si la persona no puede entrar, la app se detiene aquí
# ------------------------------------------------------------------
evitar_traduccion()
contexto = controlar_acceso()

# ------------------------------------------------------------------
# Menú lateral
# ------------------------------------------------------------------
st.sidebar.title("Predicta Retail")
st.sidebar.caption("Gestión de inventarios con Machine Learning")
seccion = st.sidebar.radio("Menú", list(SECCIONES), label_visibility="collapsed")

# ------------------------------------------------------------------
# Carga de datos (con validaciones)
# ------------------------------------------------------------------
datos = cargar_datos()

if datos["errores"]:
    st.error("No se pudieron cargar los datos. Corrige estos problemas:")
    for error in datos["errores"]:
        st.write("•", error)
    # Aunque haya errores, se permite importar archivos corregidos o regenerar los de prueba
    st.subheader("Importar archivos corregidos")
    productos.mostrar_importacion(datos)
    st.stop()

for aviso in datos["advertencias"]:
    st.sidebar.warning(aviso)

# ------------------------------------------------------------------
# Filtros (en el menú lateral)
# ------------------------------------------------------------------
fecha_min = datos["ventas"]["fecha"].min().date()
fecha_max = datos["ventas"]["fecha"].max().date()

st.sidebar.divider()
st.sidebar.subheader("Filtros")
rango = st.sidebar.date_input("Rango de fechas", value=(fecha_min, fecha_max),
                              min_value=fecha_min, max_value=fecha_max, format="DD/MM/YYYY")
# Mientras el usuario elige, el calendario puede devolver una sola fecha
if isinstance(rango, (tuple, list)) and len(rango) == 2:
    fecha_inicio, fecha_fin = rango
else:
    fecha_inicio = rango[0] if isinstance(rango, (tuple, list)) else rango
    fecha_fin = fecha_inicio

categorias = st.sidebar.multiselect("Categorías", sorted(datos["productos"]["categoria"].unique()),
                                    placeholder="Todas")
catalogo = datos["productos"]
if categorias:
    catalogo = catalogo[catalogo["categoria"].isin(categorias)]
etiquetas = dict(zip(catalogo["codigo"],
                     catalogo["codigo"] + " · " + catalogo["nombre"] + " " + catalogo["talla"] + " " + catalogo["color"]))
codigos = st.sidebar.multiselect("Productos", list(etiquetas), format_func=etiquetas.get,
                                 placeholder="Todos")

filtros = {"fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin,
           "categorias": categorias, "codigos": codigos}

st.sidebar.divider()
st.sidebar.caption(f"{'⚠️' if datos['ficticios'] else '📁'} {fuente(datos)}  \n🗄️ {origen(datos)}")
mostrar_usuario(contexto)

# ------------------------------------------------------------------
# Mostrar la sección elegida
# ------------------------------------------------------------------
SECCIONES[seccion].mostrar(datos, filtros)
