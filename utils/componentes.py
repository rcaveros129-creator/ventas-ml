"""
Piezas visuales que usan varias secciones: tarjetas, estilo de gráficos,
colores y formato de montos.
"""

import streamlit as st

AZUL = "#2a78d6"
NARANJA = "#eb6834"
AQUA = "#1baf7a"
GRIS_TEXTO = "#52514e"
GRIS_GRILLA = "#ececea"

# Colores de ESTADO (siempre acompañados de ícono y texto, nunca solos)
ESTADOS = {
    "Stock crítico": {"icono": "⛔", "color": "#d03b3b", "orden": 0},
    "Próximo a agotarse": {"icono": "⚠️", "color": "#fab219", "orden": 1},
    "Stock adecuado": {"icono": "✅", "color": "#0ca30c", "orden": 2},
    "Exceso de inventario": {"icono": "📦", "color": "#7d8590", "orden": 3},
}


def fuente(datos):
    """Texto que indica si se trabaja con datos de prueba o con datos importados."""
    return "Datos FICTICIOS de prueba" if datos.get("ficticios") else "Datos importados por el usuario"


def origen(datos):
    """De dónde se leyeron los datos: base de datos o archivos CSV."""
    return datos.get("origen", "Archivos CSV")


def etiqueta_estado(estado):
    return f"{ESTADOS[estado]['icono']} {estado}"


def formato_soles(valor):
    return f"S/ {valor:,.0f}"


def tarjeta(columna, titulo, valor, nota=""):
    """Tarjeta con título, valor grande y una nota."""
    with columna.container(border=True, height="stretch"):
        st.caption(titulo)
        st.markdown(f"### {valor}")
        if nota:
            st.caption(nota)


def tarjeta_producto(columna, titulo, fila):
    """Tarjeta con el nombre completo de un producto y sus unidades."""
    with columna.container(border=True, height="stretch"):
        st.caption(titulo)
        st.markdown(f"**{fila['producto']}**")
        st.caption(f"{fila['codigo']} · {int(fila['unidades']):,} unidades")


def estilo_grafico(fig, alto=340):
    """Estilo limpio y uniforme para los gráficos de barras de una sola serie."""
    fig.update_traces(marker_color=AZUL)
    fig.update_layout(
        template="plotly_white", height=alto, barcornerradius=4,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=GRIS_TEXTO, size=13),
        hoverlabel=dict(bgcolor="white", font_size=13),
        xaxis_title=None, yaxis_title=None,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRIS_GRILLA)
    return fig


def evitar_traduccion():
    """
    Indica al navegador que la página ya está en español, para que el
    traductor automático (Edge/Chrome) no cambie textos ni números.
    """
    st.html("""
    <script>
      const raiz = document.documentElement;
      raiz.setAttribute("lang", "es");
      raiz.setAttribute("translate", "no");
      raiz.classList.add("notranslate");
      if (!document.querySelector('meta[name="google"]')) {
        const meta = document.createElement("meta");
        meta.name = "google";
        meta.content = "notranslate";
        document.head.appendChild(meta);
      }
    </script>
    """, unsafe_allow_javascript=True)
