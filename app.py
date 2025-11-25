# app.py
# Versión refactor ligera para tu repo (usa historico_bolita_MASTER.csv y bolitero_nivel_dios_model.h5)
# Usa cache de Streamlit para acelerar cargas y evita recargar modelo innecesariamente.

import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Bolita Cubana Play Creator", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical(csv_path: Path) -> Optional[pd.DataFrame]:
    """Carga CSV histórico con caching. Devuelve None si no existe."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error al leer CSV {csv_path.name}: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_keras_model(h5_path: Path):
    """Carga un modelo Keras (.h5) si existe. Devuelve None si no se puede cargar."""
    if not h5_path.exists():
        return None
    try:
        # Import dentro de la función para no cargar tensorflow si no se necesita
        from tensorflow.keras.models import load_model
        model = load_model(str(h5_path))
        return model
    except Exception as e:
        st.warning(f"No se pudo cargar modelo Keras ({h5_path.name}): {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_joblib(path: Path):
    """Carga objetos guardados con joblib (.pkl)."""
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"No se pudo cargar {path.name}: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_stub(state: str) -> Dict[str, Any]:
    """Función placeholder para obtener resultados diarios (usar tu scraper real)."""
    return {"state": state, "numbers": [], "note": "Función stub — ajustar selectores/URLs reales"}


def main():
    st.title("Bolita Cubana Play Creator")
    st.write("Aplicación para ver historial, ejecutar el actualizador y usar tu modelo de predicción.")

    # Rutas relativas a tu repo
    csv_path = BASE_DIR / "historico_bolita_MASTER.csv"
    model_h5 = BASE_DIR / "bolitero_nivel_dios_model.h5"
    scaler_pkl = BASE_DIR / "minmax_scaler.pkl"
    encoder_pkl = BASE_DIR / "onehot_encoder.pkl"

    # Cargar recursos
    df = load_historical(csv_path)
    keras_model = load_keras_model(model_h5)
    scaler = load_joblib(scaler_pkl)
    encoder = load_joblib(encoder_pkl)

    tabs = st.tabs(["Resultados Diarios", "Predicción Numerológica", "Historial", "Actualizador"])

    with tabs[0]:
        st.header("Resultados Diarios")
        col1, col2 = st.columns([1, 2])
        with col1:
            state = st.selectbox("Selecciona Estado", ["GA", "FL", "NY"])
            if st.button("Obtener (stub)"):
                with st.spinner("Obteniendo..."):
                    res = fetch_daily_stub(state)
                    if res.get("error"):
                        st.error(res["error"])
                    else:
                        st.success(f"Resultados para {state} (stub)")
                        st.write(res)
        with col2:
            st.info("Si quieres que integremos el scraper real, pega las URLs o dime donde está el script Actualiza csv.py.")

    with tabs[1]:
        st.header("Predicción Numerológica")
        st.write("Introduce datos — el formulario es de ejemplo, ajústalo según tu pipeline de entrenamiento.")
        with st.form("pred_form"):
            numero_base = st.number_input("Número base", min_value=0, max_value=999, value=0)
            fecha = st.date_input("Fecha de referencia")
            submit = st.form_submit_button("Predecir")
        if submit:
            if keras_model is None:
                st.error("No se detectó modelo Keras. Añade bolitero_nivel_dios_model.h5 al repo o revisa carga.")
            else:
                # Ejemplo simple de transformación — reemplazar por tu pipeline real
                input_vec = [numero_base]
                # Intento de escalar si hay scaler
                try:
                    if scaler is not None:
                        import numpy as np
                        input_vec = scaler.transform([[numero_base]])[0].tolist()
                except Exception:
                    pass
                # Inferencia básica
                try:
                    import numpy as np
                    arr = np.array([input_vec])
                    preds = keras_model.predict(arr).tolist()
                    st.json({"predictions": preds})
                except Exception as e:
                    st.error(f"Error en predicción: {e}")

    with tabs[2]:
        st.header("Historial")
        if df is None:
            st.warning(f"No se encontró {csv_path.name} en la raíz del repo.")
            uploaded = st.file_uploader("Sube un CSV de historial (temporalmente)", type=["csv"])
            if uploaded:
                try:
                    df2 = pd.read_csv(uploaded)
                    st.success("CSV cargado en memoria.")
                    st.dataframe(df2.head(200))
                except Exception as e:
                    st.error(f"Error leyendo CSV subido: {e}")
        else:
            st.dataframe(df.head(300))
            st.download_button("Descargar historial (CSV)", df.to_csv(index=False), file_name="historico_export.csv")

    with tabs[3]:
        st.header("Ejecutador de Actualizador (opcional)")
        st.write("Aquí te recuerdo: ejecutar scripts desde la app ejecutará código en el servidor que corre la app. Úsalo sólo si confías en el script.")
        st.info("Si tu Actualiza csv.py modifica archivos del repo en el entorno de Streamlit, esos cambios NO se guardan en GitHub automáticamente.")
        st.write("Si quieres puedo ayudarte a integrar ese script en un GitHub Action por separado.")

    st.sidebar.header("Estado de archivos detectados")
    st.sidebar.write({
        "historico_bolita_MASTER.csv": csv_path.exists(),
        "bolitero_nivel_dios_model.h5": model_h5.exists(),
        "minmax_scaler.pkl": scaler_pkl.exists(),
        "onehot_encoder.pkl": encoder_pkl.exists(),
    })


if __name__ == "__main__":
    main()