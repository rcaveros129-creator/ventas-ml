"""
PREDICCIÓN DE DEMANDA con Machine Learning.

Resumen del método:
1. Se arma una tabla con las ventas DIARIAS de cada producto.
2. Los días en que el producto estaba AGOTADO se marcan como "sin dato":
   ese día no se vendió porque no había stock, no porque nadie lo quisiera.
3. Para cada día se calculan variables: calendario (día de la semana...),
   fechas comerciales, categoría, precio y ventas de los días ANTERIORES.
4. Se comparan 3 modelos:
     - Promedio 28 días   (referencia: "seguirá vendiendo como el último mes")
     - Promedio 7 días    (referencia: "seguirá vendiendo como la última semana")
     - Random Forest      (Machine Learning)
5. Evaluación sin mirar el futuro: el Random Forest se entrena SOLO con los
   datos anteriores a una fecha de corte y se prueba con los 90 días siguientes.
6. En cada horizonte (7, 15, 30, 60, 90 días) se usa el modelo que tuvo
   MENOR error en la evaluación.
7. Se reentrena con todo el historial y se predicen los próximos 90 días.
"""

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HORIZONTES = [7, 15, 30, 60, 90]
DIAS_PRUEBA = 90                 # últimos días reservados para evaluar
MIN_HISTORIAL = 28               # días mínimos de historia para usar una fila
RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "modelo_demanda.pkl"

MODELOS = {
    "promedio_28": "Promedio últimos 28 días",
    "promedio_7": "Promedio últimos 7 días",
    "random_forest": "Random Forest (ML)",
}

VARIABLES_BASE = [
    "dia_semana",
    "campana_escolar", "antes_dia_madre", "antes_dia_padre",
    "fiestas_patrias", "navidad", "anio_nuevo",
    "categoria_cod", "precio_venta",
    "venta_hace_7_dias", "media_7_dias", "media_28_dias", "media_91_dias",
]
# Mes y semana del año solo se usan si hay 2 años o más de historia.
# Con 1 año, cada mes aparece una sola vez (y los meses a predecir pueden no
# haberse visto nunca en el entrenamiento); en las pruebas esto empeoró el error.
VARIABLES_ESTACIONALES = ["mes", "semana_anio"]
DIAS_PARA_ESTACIONALIDAD = 730
SUAVIZADO = 0.05   # evita dividir entre cero en productos que casi no venden


def elegir_variables(dias_historial):
    if dias_historial >= DIAS_PARA_ESTACIONALIDAD:
        return VARIABLES_BASE + VARIABLES_ESTACIONALES
    return list(VARIABLES_BASE)


class ModeloRitmo:
    """
    Random Forest que predice un MULTIPLICADOR sobre el ritmo reciente:
        unidades del día = multiplicador x (promedio de los últimos 28 días)
    Así un mismo modelo sirve para productos que venden 0.1 o 3 unidades al día:
    aprende, por ejemplo, que "los sábados se vende 1.4 veces el ritmo normal".
    """

    def __init__(self, variables):
        self.variables = variables
        self.bosque = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.6,
                                            random_state=42, n_jobs=-1)

    def fit(self, tabla):
        base = tabla["media_28_dias"] + SUAVIZADO
        self.bosque.fit(tabla[self.variables], tabla["unidades"] / base)
        return self

    def predict(self, x):
        base = x["media_28_dias"].values + SUAVIZADO
        return np.clip(self.bosque.predict(x[self.variables]) * base, 0, None)

    def importancia(self):
        return pd.Series(self.bosque.feature_importances_, index=self.variables).sort_values(ascending=False)


# ------------------------------------------------------------------
# 1. Calendario comercial (fechas FIJAS del calendario peruano).
#    El modelo decide con los datos si estas fechas venden más o no.
# ------------------------------------------------------------------
def _domingo(anio, mes, numero):
    d = date(anio, mes, 1)
    while d.weekday() != 6:
        d += timedelta(days=1)
    return d + timedelta(weeks=numero - 1)


def variables_calendario(fechas):
    fechas = pd.DatetimeIndex(fechas)
    tabla = pd.DataFrame(index=fechas)
    tabla["dia_semana"] = fechas.day_of_week
    tabla["semana_anio"] = fechas.isocalendar().week.astype(int).values
    tabla["mes"] = fechas.month

    def ventana(fn):
        return np.array([fn(f.date()) for f in fechas], dtype=int)

    tabla["campana_escolar"] = ventana(lambda d: date(d.year, 2, 1) <= d <= date(d.year, 3, 15))
    tabla["antes_dia_madre"] = ventana(
        lambda d: 0 <= (_domingo(d.year, 5, 2) - d).days <= 14)
    tabla["antes_dia_padre"] = ventana(
        lambda d: 0 <= (_domingo(d.year, 6, 3) - d).days <= 14)
    tabla["fiestas_patrias"] = ventana(lambda d: date(d.year, 7, 14) <= d <= date(d.year, 7, 29))
    tabla["navidad"] = ventana(lambda d: date(d.year, 12, 10) <= d <= date(d.year, 12, 24))
    tabla["anio_nuevo"] = ventana(lambda d: date(d.year, 12, 26) <= d <= date(d.year, 12, 31))
    return tabla


# ------------------------------------------------------------------
# 2. Tabla diaria: ventas por producto y día, marcando días agotados
# ------------------------------------------------------------------
def construir_series(ventas, productos, compras):
    """
    Devuelve:
      demanda  -> tabla (fechas x productos) con unidades vendidas por día.
                  NaN = día sin dato (producto agotado o aún no lanzado).
      agotado  -> tabla booleana: True si el producto terminó el día sin stock.
    """
    fechas = pd.date_range(ventas["fecha"].min(), ventas["fecha"].max(), freq="D")
    codigos = productos["codigo"].tolist()

    vendidas = (ventas.pivot_table(index="fecha", columns="codigo", values="cantidad", aggfunc="sum")
                .reindex(index=fechas, columns=codigos).fillna(0))

    # Reconstruir el stock al final de cada día, yendo hacia atrás desde hoy:
    # stock(día) = stock actual + ventas posteriores - compras recibidas posteriores
    recibidas = compras[compras["estado"].str.lower() == "recibido"]
    recibidas = (recibidas.pivot_table(index="fecha_recepcion", columns="codigo",
                                       values="cantidad", aggfunc="sum")
                 .reindex(index=fechas, columns=codigos).fillna(0))
    stock_hoy = productos.set_index("codigo")["stock_actual"].reindex(codigos)
    ventas_posteriores = vendidas[::-1].cumsum()[::-1].shift(-1).fillna(0)
    compras_posteriores = recibidas[::-1].cumsum()[::-1].shift(-1).fillna(0)
    stock_fin_dia = stock_hoy + ventas_posteriores - compras_posteriores
    # Agotado = el stock reconstruido es exactamente 0. Un valor negativo indica que
    # faltan compras en el historial (dato incompleto), no que estuviera agotado.
    agotado = stock_fin_dia == 0
    if recibidas.values.sum() == 0:
        # Sin historial de compras no se puede reconstruir el stock pasado
        agotado = pd.DataFrame(False, index=fechas, columns=codigos)

    demanda = vendidas.mask(agotado)   # día agotado -> sin dato

    # Antes de la fecha de alta el producto no existía
    if "fecha_alta" in productos:
        altas = pd.to_datetime(productos.set_index("codigo")["fecha_alta"], errors="coerce").reindex(codigos)
        for codigo, alta in altas.items():
            if pd.notna(alta):
                demanda.loc[demanda.index < alta, codigo] = np.nan
    return demanda, agotado


# ------------------------------------------------------------------
# 3. Variables de cada fila (producto-día). Solo usan días ANTERIORES.
# ------------------------------------------------------------------
def _variables_historicas(demanda):
    """Para cada día t usa solo los días t-1, t-2, ... (nunca el mismo día ni el futuro)."""
    previa = demanda.shift(1)
    return {
        "venta_hace_7_dias": demanda.shift(7),
        "media_7_dias": previa.rolling(7, min_periods=1).mean(),
        "media_28_dias": previa.rolling(28, min_periods=7).mean(),
        "media_91_dias": previa.rolling(91, min_periods=7).mean(),
    }


def tabla_entrenamiento(demanda, productos):
    """Convierte las series en filas (producto, día) con sus variables y objetivo."""
    historicas = _variables_historicas(demanda)
    calendario = variables_calendario(demanda.index)
    info = _info_productos(productos)
    dias_existentes = demanda.notna().cumsum()   # días con dato hasta la fecha

    filas = []
    for codigo in demanda.columns:
        t = calendario.copy()
        t["codigo"] = codigo
        t["unidades"] = demanda[codigo].values
        for nombre, matriz in historicas.items():
            t[nombre] = matriz[codigo].values
        t["dias_desde_alta"] = dias_existentes[codigo].values
        t["categoria_cod"] = info.loc[codigo, "categoria_cod"]
        t["precio_venta"] = info.loc[codigo, "precio_venta"]
        filas.append(t)
    tabla = pd.concat(filas)
    tabla = tabla[tabla["unidades"].notna() & (tabla["dias_desde_alta"] >= MIN_HISTORIAL)]
    return _rellenar(tabla)


def _info_productos(productos):
    info = productos.set_index("codigo")[["categoria", "precio_venta"]].copy()
    categorias = sorted(info["categoria"].unique())
    info["categoria_cod"] = info["categoria"].map({c: i for i, c in enumerate(categorias)})
    return info


def _rellenar(tabla):
    """Si falta una variable histórica, se usa el promedio disponible más cercano."""
    tabla = tabla.copy()
    tabla["media_28_dias"] = tabla["media_28_dias"].fillna(tabla["media_7_dias"])
    tabla["media_91_dias"] = tabla["media_91_dias"].fillna(tabla["media_28_dias"])
    tabla["venta_hace_7_dias"] = tabla["venta_hace_7_dias"].fillna(tabla["media_7_dias"])
    return tabla.fillna(0)


def entrenar_random_forest(tabla, variables):
    return ModeloRitmo(variables).fit(tabla)


# ------------------------------------------------------------------
# 4. Pronóstico paso a paso (cada día predicho alimenta al siguiente)
# ------------------------------------------------------------------
def pronosticar(modelo, demanda, productos, dias):
    """
    Predice 'dias' días después del último día de 'demanda' para todos los productos.
    Devuelve una tabla (fechas futuras x productos) con unidades diarias.
    """
    info = _info_productos(productos).reindex(demanda.columns)
    serie = demanda.copy()
    existentes = demanda.notna().sum()
    futuras = pd.date_range(demanda.index[-1] + pd.Timedelta(days=1), periods=dias, freq="D")
    calendario = variables_calendario(futuras)

    for i, fecha in enumerate(futuras):
        valores = serie.values
        x = pd.DataFrame({
            "venta_hace_7_dias": valores[-7],
            "media_7_dias": _media(valores[-7:], minimo=1),
            "media_28_dias": _media(valores[-28:], minimo=7),
            "media_91_dias": _media(valores[-91:], minimo=7),
        }, index=serie.columns)
        for col in calendario.columns:
            x[col] = calendario.iloc[i][col]
        x["dias_desde_alta"] = existentes.values + i
        x["categoria_cod"] = info["categoria_cod"].values
        x["precio_venta"] = info["precio_venta"].values
        x = _rellenar(x)
        prediccion = modelo.predict(x)
        serie.loc[fecha] = prediccion
    return serie.loc[futuras]


def _media(valores, minimo):
    """Promedio por columna ignorando días sin dato; NaN si hay menos de 'minimo' datos."""
    validos = np.isfinite(valores).sum(axis=0)
    suma = np.nansum(valores, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        media = suma / validos
    media[validos < minimo] = np.nan
    return media


def pronostico_promedio(demanda, ventana, dias):
    """Modelo de referencia: repetir el promedio diario de los últimos 'ventana' días con dato."""
    tasa = demanda.iloc[-ventana:].mean()
    tasa = tasa.fillna(demanda.iloc[-91:].mean()).fillna(0)
    futuras = pd.date_range(demanda.index[-1] + pd.Timedelta(days=1), periods=dias, freq="D")
    return pd.DataFrame(np.tile(tasa.values, (dias, 1)), index=futuras, columns=demanda.columns)


# ------------------------------------------------------------------
# 5. Evaluación temporal (backtesting)
# ------------------------------------------------------------------
def evaluar(demanda, ventas_reales, productos):
    """
    Entrena con los datos ANTERIORES al corte y predice desde varios puntos de
    partida dentro de los últimos 90 días. Compara contra lo que realmente se vendió.
    Métrica: error porcentual absoluto ponderado (WAPE):
        suma |predicho - real| / suma real   (por producto y horizonte)
    """
    corte = len(demanda) - DIAS_PRUEBA
    variables = elegir_variables(corte)
    modelo = entrenar_random_forest(tabla_entrenamiento(demanda.iloc[:corte], productos), variables)

    registros = []
    for inicio in range(corte, len(demanda) - min(HORIZONTES) + 1, 15):
        restante = len(demanda) - inicio
        dias = min(max(HORIZONTES), restante)
        historia = demanda.iloc[:inicio]
        preds = {
            "random_forest": pronosticar(modelo, historia, productos, dias),
            "promedio_28": pronostico_promedio(historia, 28, dias),
            "promedio_7": pronostico_promedio(historia, 7, dias),
        }
        for h in HORIZONTES:
            if h > restante:
                continue
            real = ventas_reales.iloc[inicio:inicio + h].sum()
            for nombre, p in preds.items():
                predicho = p.iloc[:h].sum()
                registros.append({"horizonte": h, "modelo": nombre, "inicio": demanda.index[inicio],
                                  "error_abs": (predicho - real).abs().sum(), "real": real.sum(),
                                  "sesgo": (predicho - real).sum()})

    detalle = pd.DataFrame(registros)
    resumen = (detalle.groupby(["horizonte", "modelo"])
               .agg(error_abs=("error_abs", "sum"), real=("real", "sum"),
                    sesgo=("sesgo", "sum"), pruebas=("inicio", "nunique"))
               .reset_index())
    resumen["wape"] = resumen["error_abs"] / resumen["real"] * 100
    resumen["sesgo_pct"] = resumen["sesgo"] / resumen["real"] * 100
    return resumen, demanda.index[corte]


# ------------------------------------------------------------------
# 6. Proceso completo: evaluar, reentrenar, predecir y guardar
# ------------------------------------------------------------------
VERSION_MODELO = 2   # súbela si cambia el método: obliga a reentrenar el modelo guardado


def firma_datos(ventas, productos, compras):
    """Identifica la versión de los datos y del método; si cambia, hay que reentrenar."""
    return (VERSION_MODELO, len(ventas), int(ventas["cantidad"].sum()), str(ventas["fecha"].max().date()),
            len(productos), int(productos["stock_actual"].sum()), len(compras))


def ejecutar_todo(ventas, productos, compras, forzar=False):
    firma = firma_datos(ventas, productos, compras)
    if not forzar and RUTA_MODELO.exists():
        try:
            guardado = joblib.load(RUTA_MODELO)
            if guardado.get("firma") == firma:
                guardado["cargado_de_archivo"] = True
                return guardado
        except Exception:
            pass   # archivo dañado o de otra versión: se vuelve a entrenar

    demanda, agotado = construir_series(ventas, productos, compras)
    ventas_reales = (ventas.pivot_table(index="fecha", columns="codigo", values="cantidad", aggfunc="sum")
                     .reindex(index=demanda.index, columns=demanda.columns).fillna(0))

    if len(demanda) < DIAS_PRUEBA + 90:
        raise ValueError("Se necesitan al menos 6 meses de ventas para entrenar y evaluar el modelo.")

    resumen, fecha_corte = evaluar(demanda, ventas_reales, productos)
    mejor = (resumen.sort_values("wape").groupby("horizonte").first()["modelo"].to_dict())

    # Modelo final con TODO el historial
    tabla = tabla_entrenamiento(demanda, productos)
    modelo = entrenar_random_forest(tabla, elegir_variables(len(demanda)))
    dias = max(HORIZONTES)
    futuros = {
        "random_forest": pronosticar(modelo, demanda, productos, dias),
        "promedio_28": pronostico_promedio(demanda, 28, dias),
        "promedio_7": pronostico_promedio(demanda, 7, dias),
    }

    # Tabla final: para cada horizonte, la predicción del mejor modelo
    # Si dos horizontes usan modelos distintos, un horizonte largo nunca puede
    # predecir menos unidades que uno más corto (la venta acumulada no baja).
    filas = []
    for codigo in demanda.columns:
        fila, anterior = {"codigo": codigo}, 0.0
        for h in HORIZONTES:
            modelo_h = mejor[h]
            wape = resumen.query("horizonte == @h and modelo == @modelo_h")["wape"].iloc[0]
            valor = max(futuros[modelo_h][codigo].iloc[:h].sum(), anterior)
            anterior = valor
            fila[f"pred_{h}"] = valor
            fila[f"min_{h}"] = max(0, valor * (1 - wape / 100))
            fila[f"max_{h}"] = valor * (1 + wape / 100)
        filas.append(fila)

    importancia = modelo.importancia()

    resultado = {
        "firma": firma,
        "fecha_entrenamiento": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
        "ultimo_dia_datos": demanda.index[-1],
        "fecha_corte_evaluacion": fecha_corte,
        "filas_entrenamiento": len(tabla),
        "variables": modelo.variables,
        "usa_estacionalidad_anual": "mes" in modelo.variables,
        "modelo": modelo,
        "evaluacion": resumen,
        "mejor_modelo": mejor,
        "predicciones": pd.DataFrame(filas),
        "pronostico_diario": futuros,
        "dias_agotado": agotado.sum().to_dict(),
        "importancia": importancia,
        "cargado_de_archivo": False,
    }
    RUTA_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump(resultado, RUTA_MODELO, compress=3)
    return resultado
