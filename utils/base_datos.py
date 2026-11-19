"""
CONEXIÓN CON LA BASE DE DATOS (PostgreSQL en Supabase).

La conexión se configura en el archivo .streamlit/secrets.toml (en tu
computadora) o en "Secrets" de Streamlit Cloud (en la app publicada):

    [base_datos]
    url = "postgresql://postgres.xxxx:TU-CONTRASEÑA@aws-1-us-west-2.pooler.supabase.com:5432/postgres"
    tienda_id = 1

Si no hay configuración, el aplicativo sigue usando los archivos CSV de la
carpeta 'data' (útil para practicar sin internet).
"""

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Columnas que se guardan en cada tabla (las demás se calculan al leer)
COLUMNAS_TABLA = {
    "productos": ["codigo", "nombre", "categoria", "talla", "color", "precio_compra", "precio_venta",
                  "stock_actual", "stock_minimo", "tiempo_reposicion_dias", "proveedor", "fecha_alta"],
    "ventas": ["id_venta", "fecha", "codigo", "cantidad", "precio_compra", "precio_venta", "total_venta"],
    "compras": ["id_compra", "fecha_pedido", "fecha_recepcion", "codigo", "proveedor", "cantidad",
                "costo_unitario", "costo_total", "estado"],
}
FECHAS = {"productos": ["fecha_alta"], "ventas": ["fecha"], "compras": ["fecha_pedido", "fecha_recepcion"]}


def configuracion():
    """Devuelve la sección [base_datos] de los secretos, o None si no existe."""
    try:
        return dict(st.secrets["base_datos"])
    except Exception:  # no hay archivo de secretos o no tiene la sección
        return None


def configurada():
    config = configuracion()
    return bool(config and config.get("url"))


def tienda_id():
    """
    Tienda con la que se trabaja: la del usuario que inició sesión (guardada en
    su sesión al entrar) o, sin inicio de sesión, la indicada en secrets.toml.
    """
    from streamlit import runtime
    if runtime.exists() and st.session_state.get("tienda_id") is not None:
        return int(st.session_state["tienda_id"])
    return int((configuracion() or {}).get("tienda_id", 1))


def _armar_url(texto):
    """
    Convierte la dirección copiada de Supabase en una URL segura para SQLAlchemy.
    Acepta contraseñas con símbolos (@, #, /...) y quita los corchetes si se copiaron.
    """
    texto = texto.strip()
    if "://" not in texto or "@" not in texto:
        raise ValueError("La dirección de la base de datos no tiene el formato postgresql://usuario:clave@host:puerto/base")
    cuerpo = texto.split("://", 1)[1]
    credenciales, servidor = cuerpo.rsplit("@", 1)
    usuario, clave = credenciales.split(":", 1)
    clave = clave.strip("[]")
    if clave.upper() == "YOUR-PASSWORD":
        raise ValueError("Falta reemplazar [YOUR-PASSWORD] por la contraseña real de la base de datos.")
    host_puerto, _, base = servidor.partition("/")
    host, _, puerto = host_puerto.partition(":")
    return URL.create("postgresql+psycopg", username=usuario, password=clave, host=host,
                      port=int(puerto or 5432), database=(base.split("?")[0] or "postgres"))


@st.cache_resource(show_spinner=False)
def motor():
    """Conexión reutilizable con la base de datos (se crea una sola vez)."""
    return create_engine(_armar_url(configuracion()["url"]), pool_pre_ping=True, pool_size=2, max_overflow=2)


def buscar_usuario(email):
    """Busca un correo en la tabla 'usuarios'. Devuelve sus datos y su tienda, o None."""
    consulta = text("""
        select u.email, u.nombre, u.rol, u.activo, u.tienda_id, t.nombre as tienda
        from usuarios u join tiendas t on t.id = u.tienda_id
        where u.email = :e""")
    with motor().connect() as conexion:
        fila = conexion.execute(consulta, {"e": email.strip().lower()}).mappings().first()
    return dict(fila) if fila else None


def leer_tienda(id_tienda):
    """Lee la tienda y sus 3 tablas. Devuelve (info_tienda, {tipo: DataFrame})."""
    with motor().connect() as conexion:
        tienda = conexion.execute(text("select id, nombre, datos_ficticios from tiendas where id = :t"),
                                  {"t": id_tienda}).mappings().first()
        if tienda is None:
            raise ValueError(f"No existe la tienda con id {id_tienda} en la base de datos.")
        tablas = {}
        for tipo, columnas in COLUMNAS_TABLA.items():
            consulta = text(f"select {', '.join(columnas)} from {tipo} where tienda_id = :t")
            tablas[tipo] = pd.read_sql(consulta, conexion, params={"t": id_tienda}, parse_dates=FECHAS[tipo])
    return dict(tienda), tablas


def reemplazar_tabla(tipo, tabla, id_tienda, ficticios=None):
    """
    Reemplaza TODAS las filas de una tabla para una tienda (en una sola
    transacción: si algo falla, no se borra nada).
    """
    columnas = [c for c in COLUMNAS_TABLA[tipo] if c in tabla.columns]
    datos = tabla[columnas].copy()
    for col in FECHAS[tipo]:
        if col in datos:
            datos[col] = pd.to_datetime(datos[col], errors="coerce").dt.date
    datos = datos.astype(object).where(datos.notna(), None)   # NaN -> NULL
    datos["tienda_id"] = id_tienda
    with motor().begin() as conexion:
        conexion.execute(text(f"delete from {tipo} where tienda_id = :t"), {"t": id_tienda})
        datos.to_sql(tipo, conexion, if_exists="append", index=False, method="multi", chunksize=1000)
        if ficticios is not None:
            conexion.execute(text("update tiendas set datos_ficticios = :f where id = :t"),
                             {"f": ficticios, "t": id_tienda})
    return len(datos)


def exportar_tabla(tipo, id_tienda):
    """Copia actual de una tabla de la tienda (para respaldo antes de reemplazarla)."""
    with motor().connect() as conexion:
        consulta = text(f"select {', '.join(COLUMNAS_TABLA[tipo])} from {tipo} where tienda_id = :t")
        return pd.read_sql(consulta, conexion, params={"t": id_tienda})
