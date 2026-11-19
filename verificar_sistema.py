"""
PREDICTA RETAIL - Verificación completa del sistema (Etapa 9).

Revisa, de principio a fin, que los datos, los modelos, los cálculos y las
8 secciones de la interfaz funcionen juntos. Ejecútalo con el entorno virtual
activado:

    python verificar_sistema.py

Tarda alrededor de 1 a 2 minutos. Al final muestra un resumen con ✅ y ❌.
"""

import sys
import time
import warnings
from io import BytesIO

import streamlit.logger

streamlit.logger.set_log_level("error")       # ocultar avisos internos de Streamlit
warnings.filterwarnings("ignore")

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402

resultados = []


def revisar(nombre):
    """Decorador: ejecuta una revisión y anota si pasó o falló."""
    def envoltura(funcion):
        inicio = time.time()
        try:
            detalle = funcion()
            resultados.append((True, nombre, detalle or ""))
            print(f"✅ {nombre}  {detalle or ''}  ({time.time() - inicio:.1f} s)")
        except Exception as e:  # cualquier fallo se reporta, sin detener las demás revisiones
            resultados.append((False, nombre, str(e)))
            print(f"❌ {nombre}\n     → {e}")
        return funcion
    return envoltura


print("=" * 70)
print("VERIFICACIÓN COMPLETA DE PREDICTA RETAIL")
print("=" * 70)

from utils.data_loader import cargar_datos          # noqa: E402

datos = cargar_datos()
contexto = {}


@revisar("1. Archivos de datos cargados y validados")
def _():
    assert not datos["errores"], " | ".join(datos["errores"])
    v, p, c = datos["ventas"], datos["productos"], datos["compras"]
    origen = "ficticios" if datos["ficticios"] else "importados"
    return f"{len(p)} productos, {len(v):,} ventas, {len(c)} compras (datos {origen}) · {datos['origen']}"


@revisar("2. Coherencia entre archivos")
def _():
    v, p, c = datos["ventas"], datos["productos"], datos["compras"]
    assert set(v["codigo"]) <= set(p["codigo"]), "Hay ventas de productos que no existen"
    assert set(c["codigo"]) <= set(p["codigo"]), "Hay compras de productos que no existen"
    assert (v["cantidad"] > 0).all(), "Hay ventas con cantidad 0 o negativa"
    assert (p["stock_actual"] >= 0).all(), "Hay stock negativo"
    assert (c["fecha_recepcion"] >= c["fecha_pedido"]).all(), "Hay compras recibidas antes de pedirse"
    return "códigos, cantidades, stock y fechas correctos"


@revisar("3. Rotación con K-Means")
def _():
    from utils import ml_rotacion
    v = datos["ventas"]
    variables = ml_rotacion.calcular_variables(v, datos["productos"], v["fecha"].min(), v["fecha"].max())
    tabla, calidad, _ = ml_rotacion.clasificar(variables)
    clases = set(tabla["rotacion"])
    assert clases == set(ml_rotacion.CLASES), f"Clases encontradas: {clases}"
    assert calidad > 0.25, f"Grupos poco separados (silhouette {calidad:.2f})"
    conteo = tabla["rotacion"].value_counts().reindex(ml_rotacion.CLASES).tolist()
    return f"alta/media/baja = {conteo}, silhouette {calidad:.2f}"


@revisar("4. Predicción de demanda (entrenamiento, evaluación y guardado)")
def _():
    from utils import ml_demanda
    res = ml_demanda.ejecutar_todo(datos["ventas"], datos["productos"], datos["compras"])
    contexto["res"] = res
    pred = res["predicciones"]
    assert ml_demanda.RUTA_MODELO.exists(), "No se guardó models/modelo_demanda.pkl"
    columnas = [f"pred_{h}" for h in ml_demanda.HORIZONTES]
    assert (pred[columnas] >= 0).all().all(), "Hay predicciones negativas"
    assert (np.diff(pred[columnas].values, axis=1) >= -1e-9).all(), "Un horizonte largo predice menos que uno corto"
    ev = res["evaluacion"]
    wape_30 = ev[(ev["horizonte"] == 30) & (ev["modelo"] == res["mejor_modelo"][30])]["wape"].iloc[0]
    usados = sorted(set(ml_demanda.MODELOS[m] for m in res["mejor_modelo"].values()))
    return f"error a 30 días {wape_30:.1f} %, modelo(s) usado(s): {', '.join(usados)}"


@revisar("5. Sin información del futuro en el entrenamiento")
def _():
    from utils import ml_demanda
    demanda, _ = ml_demanda.construir_series(datos["ventas"], datos["productos"], datos["compras"])
    tabla = ml_demanda.tabla_entrenamiento(demanda, datos["productos"])
    revisadas = 0
    for _, fila in tabla.sample(200, random_state=1).iterrows():
        dia = fila.name
        previos = demanda.loc[:dia - pd.Timedelta(days=1), fila["codigo"]].iloc[-7:]
        if previos.notna().any():
            assert abs(previos.mean() - fila["media_7_dias"]) < 1e-9, f"Fuga de información en {dia:%d/%m/%Y}"
            revisadas += 1
    res = contexto.get("res")
    if res is not None:
        assert res["fecha_corte_evaluacion"] < demanda.index[-1], "La evaluación no separa pasado y futuro"
    return f"{revisadas} filas revisadas: solo usan días anteriores"


@revisar("6. Control predictivo de stock")
def _():
    from utils import stock
    caso = stock.evaluar_caso(20, 4, 7)
    assert caso["estado"] == "Stock crítico", f"Ejemplo 20 unid. / 4 por día / 7 días dio: {caso['estado']}"
    assert abs(caso["dias_hasta_agotar"] - 5) < 1e-9, "El ejemplo debería durar 5 días"
    tabla = stock.calcular_stock(datos["productos"], datos["ventas"], datos["compras"], contexto["res"])
    contexto["stock"] = tabla
    assert tabla["estado"].isin(["Stock crítico", "Próximo a agotarse", "Stock adecuado",
                                 "Exceso de inventario"]).all()
    assert (tabla["cantidad_reponer"] >= 0).all(), "Cantidades a reponer negativas"
    conteo = tabla["estado"].value_counts().to_dict()
    return "ejemplo del documento → crítico ✓; " + ", ".join(f"{k}: {v}" for k, v in conteo.items())


@revisar("7. Recomendaciones de abastecimiento")
def _():
    from utils.abastecimiento import recomendar
    plan = recomendar(contexto["stock"], datos["productos"], contexto["res"])
    contexto["plan"] = plan
    hoy = pd.Timestamp(contexto["res"]["ultimo_dia_datos"])
    pedidos = plan[plan["fecha_pedido"].notna()]
    assert (pedidos["fecha_pedido"] > hoy).all(), "Hay pedidos con fecha en el pasado"
    assert (pedidos["cantidad"] >= 1).all(), "Hay pedidos de 0 unidades"
    assert (pedidos["fecha_llegada"] > pedidos["fecha_pedido"]).all()
    return f"{len(pedidos)} pedidos programados, {(plan['accion'] == 'No comprar: exceso').sum()} con exceso"


@revisar("8. Análisis de temporadas")
def _():
    from utils import temporadas
    res = temporadas.analizar_temporadas(datos["ventas"]).set_index("temporada")
    assert not res.empty, "No se encontraron temporadas completas"
    texto = f"{len(res)} temporadas analizadas"
    if datos["ficticios"]:
        # En la simulación el Día del Padre NO tiene efecto y Navidad SÍ
        assert res.loc["Día del Padre", "conclusion"] == "Sin efecto comprobado", "Detectó un efecto inexistente"
        assert res.loc["Navidad", "conclusion"] == "Aumenta las ventas", "No detectó el efecto de Navidad"
        texto += "; Día del Padre sin efecto ✓, Navidad aumenta ✓"
    return texto


@revisar("9. Reporte en Excel")
def _():
    from utils import reportes as rep
    tablas = {"Ventas por producto": rep.ventas_por_producto(datos["ventas"], datos["productos"]),
              "Estado del inventario": rep.inventario(contexto["stock"]),
              "Abastecimiento": rep.abastecimiento(contexto["plan"]),
              "Predicciones": rep.predicciones(contexto["res"], datos["productos"])}
    contenido = rep.generar_excel(tablas, [("Prueba", "OK")])
    hojas = pd.read_excel(BytesIO(contenido), sheet_name=None)
    assert list(hojas) == ["Información"] + list(tablas), f"Hojas: {list(hojas)}"
    return f"{len(hojas)} hojas, {len(contenido) / 1024:.0f} KB"


@revisar("10. Interfaz: las 8 secciones se abren sin errores")
def _():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("app.py", default_timeout=300).run()
    assert not app.exception, app.exception[0].value
    menu = app.sidebar.radio[0]
    for seccion in menu.options:
        menu.set_value(seccion).run()
        assert not app.exception, f"{seccion}: {app.exception[0].value}"
    return f"{len(menu.options)} secciones revisadas"


print("=" * 70)
correctas = sum(1 for ok, _, _ in resultados if ok)
print(f"RESULTADO: {correctas} de {len(resultados)} revisiones correctas")
if correctas == len(resultados):
    print("🎉 El sistema funciona de principio a fin.")
else:
    print("Revisa los puntos marcados con ❌ (el mensaje indica qué falló).")
sys.exit(0 if correctas == len(resultados) else 1)
