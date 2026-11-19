"""
PREDICTA RETAIL - Fase A: copiar los archivos CSV a la base de datos.

Lee data/productos.csv, data/ventas.csv y data/compras.csv, los valida y los
guarda en la base de datos configurada en .streamlit/secrets.toml, dentro de
la tienda indicada en 'tienda_id'. Si la tienda ya tenía datos, se reemplazan.

Ejecútalo con el entorno virtual activado:
    python migrar_a_base_datos.py
"""

import sys
import warnings

import streamlit.logger

streamlit.logger.set_log_level("error")
warnings.filterwarnings("ignore")

from utils import base_datos                                     # noqa: E402
from utils.data_loader import CARPETA_DATOS, COLUMNAS, leer_archivo, validar_tabla  # noqa: E402

print("=" * 60)
print("COPIAR LOS DATOS CSV A LA BASE DE DATOS")
print("=" * 60)

if not base_datos.configurada():
    print("❌ No se encontró la configuración de la base de datos.")
    print("   Crea el archivo .streamlit/secrets.toml con la sección [base_datos] (ver instrucciones).")
    sys.exit(1)

try:
    id_tienda = base_datos.tienda_id()
    tienda, _ = base_datos.leer_tienda(id_tienda)
    print(f"✅ Conexión correcta. Tienda {id_tienda}: {tienda['nombre']}")
except Exception as e:
    print(f"❌ No se pudo conectar: {e}")
    print("   Revisa la dirección y la contraseña en .streamlit/secrets.toml, y tu conexión a internet.")
    sys.exit(1)

ficticios = (CARPETA_DATOS / "simulacion_referencia.csv").exists()
tablas = {}
for tipo in COLUMNAS:
    ruta = CARPETA_DATOS / f"{tipo}.csv"
    if not ruta.exists():
        print(f"❌ Falta el archivo data/{tipo}.csv")
        sys.exit(1)
    tabla, errores, _ = validar_tabla(leer_archivo(ruta), tipo)
    if errores:
        print(f"❌ {tipo}.csv tiene errores: {'; '.join(errores)}")
        sys.exit(1)
    tablas[tipo] = tabla

# Productos primero (las ventas y compras se refieren a ellos)
for tipo in ["productos", "ventas", "compras"]:
    filas = base_datos.reemplazar_tabla(tipo, tablas[tipo], id_tienda, ficticios=ficticios)
    print(f"✅ {tipo:<10} {filas:>6,} filas guardadas")

_, guardado = base_datos.leer_tienda(id_tienda)
for tipo, tabla in guardado.items():
    if len(tabla) != len(tablas[tipo]):
        print(f"❌ {tipo}: se esperaban {len(tablas[tipo])} filas y hay {len(tabla)}")
        sys.exit(1)
print("=" * 60)
print("🎉 Datos copiados y comprobados. Ahora ejecuta: streamlit run app.py")
