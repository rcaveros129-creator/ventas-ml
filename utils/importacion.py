"""
IMPORTACIÓN de archivos CSV o Excel con validaciones.

Flujo: el usuario sube un archivo -> se valida -> se muestra un resumen ->
si lo confirma, se guarda en la carpeta 'data' (con copia de respaldo del
archivo anterior) y el sistema vuelve a cargar los datos y reentrenar.
"""

from datetime import datetime

import pandas as pd

from utils import base_datos
from utils.data_loader import CARPETA_DATOS, COLUMNAS, leer_archivo, validar_tabla

CARPETA_RESPALDOS = CARPETA_DATOS / "respaldos"

DESCRIPCION = {
    "productos": "Catálogo y stock ACTUAL de cada producto (una fila por producto).",
    "ventas": "Historial de ventas (una fila por venta o por producto y día).",
    "compras": "Pedidos al proveedor: recibidos y en camino.",
}
OPCIONALES = {
    "productos": ["proveedor", "fecha_alta"],
    "ventas": ["id_venta", "precio_compra", "total_venta"],
    "compras": ["id_compra", "proveedor", "costo_total"],
}


def plantilla(tipo):
    """Archivo de ejemplo con las columnas correctas y una fila de muestra."""
    ejemplos = {
        "productos": {"codigo": "POL-001", "nombre": "Polo básico algodón", "categoria": "Polos",
                      "talla": "M", "color": "Negro", "precio_compra": 12, "precio_venta": 29,
                      "stock_actual": 40, "stock_minimo": 10, "tiempo_reposicion_dias": 5,
                      "proveedor": "Proveedor A", "fecha_alta": "2025-01-15"},
        "ventas": {"fecha": "2026-11-17", "codigo": "POL-001", "cantidad": 2, "precio_venta": 29},
        "compras": {"fecha_pedido": "2026-11-10", "fecha_recepcion": "2026-11-15", "codigo": "POL-001",
                    "cantidad": 30, "costo_unitario": 12, "estado": "Recibido"},
    }
    return pd.DataFrame([ejemplos[tipo]])


def validar_archivo(archivo, tipo, productos_actuales=None):
    """
    Lee y valida un archivo subido.
    Devuelve (tabla, errores, advertencias, resumen).
    """
    try:
        tabla = leer_archivo(archivo)
    except Exception as e:
        return None, [f"No se pudo leer el archivo: {e}"], [], {}

    tabla, errores, advertencias = validar_tabla(tabla, tipo)
    resumen = {"Filas": len(tabla), "Columnas": len(tabla.columns)}
    if errores:
        return tabla, errores, advertencias, resumen

    extra = [c for c in tabla.columns if c not in COLUMNAS[tipo] and c not in OPCIONALES[tipo]
             and c not in ("nombre", "categoria", "talla", "color")]
    if extra:
        advertencias.append(f"Columnas que el sistema no usa (se guardarán igual): {', '.join(extra)}.")

    if tipo == "ventas":
        dias = (tabla["fecha"].max() - tabla["fecha"].min()).days + 1
        resumen.update({"Desde": f"{tabla['fecha'].min():%d/%m/%Y}", "Hasta": f"{tabla['fecha'].max():%d/%m/%Y}",
                        "Días": dias, "Unidades": int(tabla["cantidad"].sum())})
        if dias < 180:
            advertencias.append(f"Solo hay {dias} días de ventas: la predicción de demanda necesita al "
                                "menos 6 meses (180 días) y funciona mejor con 1 año o más.")
    if tipo == "productos":
        resumen.update({"Productos": tabla["codigo"].nunique(), "Categorías": tabla["categoria"].nunique()})
        if len(tabla) < 6:
            advertencias.append("Con menos de 6 productos no se puede calcular la rotación con K-Means.")
    if tipo == "compras":
        resumen.update({"En camino": int((tabla["estado"].str.lower() != "recibido").sum())})

    if tipo in ("ventas", "compras") and productos_actuales is not None and "codigo" in productos_actuales:
        desconocidos = set(tabla["codigo"]) - set(productos_actuales["codigo"])
        if desconocidos:
            advertencias.append(f"{len(desconocidos)} códigos no existen en el catálogo actual "
                                f"({', '.join(sorted(desconocidos)[:5])}). Esas filas se ignorarán "
                                "a menos que también importes un catálogo de productos que los incluya.")
    return tabla, errores, advertencias, resumen


def guardar(tabla, tipo, ficticios=False):
    """
    Guarda la tabla importada:
    - Con base de datos: reemplaza las filas de la tienda (respaldo previo en data/respaldos/).
    - Sin base de datos: la guarda como data/<tipo>.csv (respaldo del archivo anterior).
    Devuelve (descripción del destino, ruta del respaldo o None).
    """
    CARPETA_RESPALDOS.mkdir(parents=True, exist_ok=True)
    marca = f"{datetime.now():%Y%m%d_%H%M%S}"

    if base_datos.configurada():
        id_tienda = base_datos.tienda_id()
        respaldo = CARPETA_RESPALDOS / f"{tipo}_tienda{id_tienda}_{marca}.csv"
        base_datos.exportar_tabla(tipo, id_tienda).to_csv(respaldo, index=False, encoding="utf-8-sig")
        filas = base_datos.reemplazar_tabla(tipo, tabla, id_tienda, ficticios=ficticios)
        return f"la base de datos ({filas} filas en '{tipo}')", respaldo

    destino = CARPETA_DATOS / f"{tipo}.csv"
    respaldo = None
    if destino.exists():
        respaldo = CARPETA_RESPALDOS / f"{tipo}_{marca}.csv"
        destino.replace(respaldo)
    excel = CARPETA_DATOS / f"{tipo}.xlsx"
    if excel.exists():   # evitar que quede un Excel viejo con el mismo nombre
        excel.replace(CARPETA_RESPALDOS / f"{tipo}_{marca}.xlsx")
    tabla = tabla.copy()
    for col in tabla.columns:
        if pd.api.types.is_datetime64_any_dtype(tabla[col]):
            tabla[col] = tabla[col].dt.strftime("%Y-%m-%d")
    tabla.to_csv(destino, index=False, encoding="utf-8-sig")
    # Con datos reales, el archivo de la simulación ya no corresponde: se archiva
    referencia = CARPETA_DATOS / "simulacion_referencia.csv"
    if referencia.exists() and not ficticios:
        referencia.replace(CARPETA_RESPALDOS / f"simulacion_referencia_{marca}.csv")
    return f"`{destino.name}`", respaldo


def cargar_datos_de_prueba():
    """
    Genera los datos ficticios (CSV) y, si hay base de datos, los copia a la tienda.
    Devuelve un texto con el resultado.
    """
    import generar_datos   # se importa aquí porque está en la carpeta principal del proyecto
    generar_datos.main()
    if not base_datos.configurada():
        return "Datos ficticios regenerados en la carpeta data."
    for tipo in COLUMNAS:
        tabla, errores, _ = validar_tabla(leer_archivo(CARPETA_DATOS / f"{tipo}.csv"), tipo)
        if errores:
            raise ValueError("; ".join(errores))
        guardar(tabla, tipo, ficticios=True)
    return f"Datos ficticios cargados en la base de datos (tienda {base_datos.tienda_id()})."
