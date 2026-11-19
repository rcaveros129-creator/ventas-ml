"""
INICIO DE SESIÓN con Google (Fase B).

- Se activa solo si existe la sección [auth] en los secretos.
- Quien entra con Google debe tener su correo en la tabla 'usuarios' y estar
  activo; si no, ve el mensaje "No tienes acceso".
- El correo decide la TIENDA que ve la persona y su ROL:
    administrador -> todo (en fases siguientes: crear tiendas y usuarios)
    dueño         -> su tienda completa, incluida la importación de datos
    empleado      -> ve su tienda, pero no puede importar ni reemplazar datos
- Sin [auth] (por ejemplo, practicando en tu computadora) no se pide inicio de
  sesión y se usa la tienda de secrets.toml.
"""

import streamlit as st

from utils import base_datos

ROLES_QUE_EDITAN = ("administrador", "dueño")


def login_configurado():
    try:
        auth = st.secrets["auth"]
        return bool(auth.get("client_id") and auth.get("client_secret"))
    except Exception:  # no hay secretos o no tienen la sección [auth]
        return False


def resolver_acceso(email):
    """
    Decide si un correo puede entrar. Devuelve (contexto, mensaje_de_error).
    contexto = {email, nombre, rol, tienda_id, tienda} cuando puede entrar.
    """
    if not email:
        return None, "Tu cuenta de Google no compartió un correo electrónico."
    usuario = base_datos.buscar_usuario(email)
    if usuario is None:
        return None, (f"El correo **{email}** no está registrado en Predicta Retail. "
                      "Pide al administrador que te dé acceso.")
    if not usuario["activo"]:
        return None, f"El acceso de **{email}** está desactivado. Consulta con el administrador."
    return usuario, None


def controlar_acceso():
    """
    Se llama al inicio de la app. Si la persona no puede entrar, muestra la
    pantalla correspondiente y detiene la app (st.stop). Si puede, guarda su
    tienda y rol en la sesión y devuelve su contexto.
    """
    if not login_configurado():
        contexto = {"email": None, "nombre": None, "rol": "administrador",
                    "tienda_id": None, "tienda": None, "con_login": False}
        st.session_state["rol"] = contexto["rol"]
        return contexto

    if not base_datos.configurada():
        st.error("El inicio de sesión necesita la base de datos: falta la sección [base_datos] en los secretos.")
        st.stop()

    if not st.user.get("is_logged_in", False):
        pantalla_de_ingreso()
        st.stop()

    email = (st.user.get("email") or "").strip().lower()
    try:
        usuario, error = resolver_acceso(email)
    except Exception as e:  # sin conexión con la base de datos
        st.error(f"No se pudo comprobar tu acceso: {e}")
        st.stop()

    if error:
        st.title("Predicta Retail")
        st.error(f"🔒 {error}")
        if st.button("Cerrar sesión e intentar con otra cuenta"):
            st.logout()
        st.stop()

    st.session_state["tienda_id"] = usuario["tienda_id"]
    st.session_state["rol"] = usuario["rol"]
    return {**usuario, "con_login": True}


def pantalla_de_ingreso():
    st.title("Predicta Retail")
    st.subheader("Gestión de inventarios con Machine Learning")
    st.write("Inicia sesión con tu cuenta de Google para ver los datos de tu tienda.")
    if st.button("Iniciar sesión con Google", type="primary"):
        st.login()
    st.caption("Solo pueden entrar los correos registrados por el administrador.")


def puede_editar():
    """True si el rol de la sesión puede importar o reemplazar datos."""
    from streamlit import runtime
    if not runtime.exists():   # scripts fuera de la app
        return True
    return st.session_state.get("rol", "administrador") in ROLES_QUE_EDITAN


def mostrar_usuario(contexto):
    """Datos del usuario y botón de cerrar sesión en el menú lateral."""
    if not contexto["con_login"]:
        return
    st.sidebar.divider()
    st.sidebar.caption(f"👤 {contexto['email']}  \n🏪 {contexto['tienda']} · {contexto['rol']}")
    if st.sidebar.button("Cerrar sesión"):
        st.logout()
