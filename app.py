"""Streamlit app for Bolita Cubana - improved.
Loads historico_bolita_MASTER.csv and model files, shows results from CSV,
allows running the Actualiza csv.py updater (in-memory), and runs predictions
without manual input using the trained Keras model and preprocessing objects.
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import subprocess
import sys
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).parent
st.set_page_config(page_title="Bolita Cubana Play Creator — Mejorada", layout="wide")

# ------------------ Loading utilities with caching ------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_historical(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure Fecha column exists and is datetime
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    # Remove rows without Fecha
    df = df.dropna(subset=["Fecha"]) if "Fecha" in df.columns else df
    # Sort
    if "Fecha" in df.columns and "Horario" in df.columns:
        df = df.sort_values(by=["Fecha", "Horario"])
    return df

@st.cache_resource(show_spinner=False)
def load_keras_model(h5_path: Path):
    if not h5_path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model
        model = load_model(str(h5_path))
        return model
    except Exception as e:
        st.warning(f"No se pudo cargar modelo Keras ({h5_path.name}): {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_joblib(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"No se pudo cargar {path.name}: {e}")
        return None

# ------------------ Prediction helpers (adapted from your script) ------------------
SEQUENCE_LENGTH = 7
NUMERICAL_FEATURES = ["Centena", "Fijo", "Corrido_1", "Corrido_2"]
CATEGORICAL_FEATURES = ["Estado", "Horario"]

# Map of game order — adjust if your dataset uses different codes
SEQUENCE_MAP = {
    ("GA", "Midday"): ("FL", "Midday"),
    ("FL", "Midday"): ("NY", "Midday"),
    ("NY", "Midday"): ("GA", "Evening"),
    ("GA", "Evening"): ("FL", "Evening"),
    ("FL", "Evening"): ("NY", "Evening"),
    ("NY", "Evening"): ("GA", "Night"),
    ("GA", "Night"): ("GA", "Midday"),
}

def obtener_proximo_sorteo(ultimo_estado, ultimo_horario):
    return SEQUENCE_MAP.get((ultimo_estado, ultimo_horario), (None, None))

def prepare_latest_sequence(df_master, sequence_length, scaler_num, encoder_cat):
    df_sequence = df_master.tail(sequence_length).copy()
    if len(df_sequence) < sequence_length:
        raise ValueError("No hay suficientes sorteos históricos para preparar la secuencia.")
    # Transform numerical
    X_num = df_sequence[NUMERICAL_FEATURES].copy()
    if scaler_num is not None:
        X_num = scaler_num.transform(X_num)
    # Encode categorical
    X_cat = None
    if encoder_cat is not None:
        enc = encoder_cat.transform(df_sequence[CATEGORICAL_FEATURES])
        enc_df = pd.DataFrame(enc, columns=encoder_cat.get_feature_names_out(CATEGORICAL_FEATURES))
        X_cat = enc_df.values
    # Combine
    if X_cat is not None:
        X_full = np.hstack([X_num, X_cat])
    else:
        X_full = X_num
    X_input = X_full.reshape(1, sequence_length, X_full.shape[1])
    return X_input

def generar_jugada_estrategica_pro(model, latest_sequence, prob_objetivo=0.80, max_numeros=20):
    predictions = model.predict(latest_sequence)
    try:
        fijo_probs = predictions[0][0]
    except Exception:
        fijo_probs = np.array(predictions[0]).ravel()
    fijo_ranking_indices = np.argsort(fijo_probs)[::-1]

    jugada_fijo_duro = []
    probabilidad_acumulada = 0.0

    for idx in fijo_ranking_indices:
        prob = float(fijo_probs[idx])
        jugada_fijo_duro.append(int(idx))
        probabilidad_acumulada += prob
        if probabilidad_acumulada >= prob_objetivo and len(jugada_fijo_duro) >= 10:
            break

    top_terminal = int(np.argmax(predictions[1][0])) if len(predictions) > 1 else None
    top_decena = int(np.argmax(predictions[2][0])) if len(predictions) > 2 else None

    top_fijos_parlet = [f for f in jugada_fijo_duro[:5]]
    parlet_plan = []
    for i in range(len(top_fijos_parlet)):
        for j in range(i + 1, len(top_fijos_parlet)):
            parlet_plan.append(f"{top_fijos_parlet[i]:02d} x {top_fijos_parlet[j]:02d}")
    if top_fijos_parlet and (top_decena is not None or top_terminal is not None):
        parlet_plan.append(f"COBERTURA DURA: {top_fijos_parlet[0]:02d} x D{top_decena} o T{top_terminal}")

    plan_de_juego = {
        "Probabilidad Acumulada (ROI)": f"{min(probabilidad_acumulada * 100, 100):.2f} %",
        "Fijos (Top 10)": [f"{n:02d}" for n in jugada_fijo_duro[:10]],
        "Fijos (Cobertura)": [f"{n:02d}" for n in jugada_fijo_duro[10:]],
        "Total de Números Únicos": len(jugada_fijo_duro),
        "Plan Parlet": parlet_plan,
        "Top Terminal": top_terminal,
        "Top Decena": top_decena,
    }
    return plan_de_juego

# ------------------ App UI ------------------
def run_updater_script(script_path: Path) -> Dict[str, Any]:
    """Run Actualiza csv.py in a subprocess and return status. Updates in-container CSV only."""
    if not script_path.exists():
        return {"ok": False, "error": f"Script no encontrado: {script_path}"}
    try:
        completed = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, timeout=300)
        return {"ok": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def main():
    st.title("Bolita Cubana Play Creator — Mejorada")
    csv_path = BASE_DIR / "historico_bolita_MASTER.csv"
    model_h5 = BASE_DIR / "bolitero_nivel_dios_model.h5"
    scaler_pkl = BASE_DIR / "minmax_scaler.pkl"
    encoder_pkl = BASE_DIR / "onehot_encoder.pkl"

    df = load_historical(csv_path)
    keras_model = load_keras_model(model_h5)
    scaler = load_joblib(scaler_pkl)
    encoder = load_joblib(encoder_pkl)

    st.sidebar.header("Controles")
    if st.sidebar.button("Actualizar CSV (ejecutar Actualiza csv.py)"):
        with st.spinner("Ejecutando actualizador... esto puede tardar segundos"):
            res = run_updater_script(BASE_DIR / "Actualiza csv.py")
            if res.get("ok"):
                try:
                    load_historical.clear()
                except Exception:
                    pass
                st.sidebar.success("Actualizador ejecutado. Recargando datos...")
                df = load_historical(csv_path)
            else:
                st.sidebar.error("Falló el actualizador. Revisa el log abajo.")
                st.sidebar.text(res.get("stderr") or res.get("error"))

    tabs = st.tabs(["Resultados", "Predicción (Automática)", "Historial"])

    # --- Resultados tab ---
    with tabs[0]:
        st.header("Resultados — usar CSV histórico")
        if df is None or df.empty:
            st.warning("No se encontró o CSV histórico está vacío. Ejecuta el actualizador o sube el CSV.")
        else:
            df['Estado'] = df['Estado'].astype(str)
            df['Horario'] = df['Horario'].astype(str)
            combos = df.groupby(['Estado', 'Horario']).size().reset_index()[['Estado', 'Horario']]
            combo_labels = [f"{e} | {h}" for e, h in zip(combos['Estado'], combos['Horario'])]
            choice = st.selectbox("Selecciona Lotería (Estado | Horario)", options=combo_labels)
            sel_estado, sel_horario = [x.strip() for x in choice.split("|")]

            df_lottery = df[(df['Estado'] == sel_estado) & (df['Horario'] == sel_horario)].copy()
            if df_lottery.empty:
                st.warning("No hay registros para esa combinación.")
            else:
                min_date = df_lottery['Fecha'].min().date()
                max_date = df_lottery['Fecha'].max().date()
                selected_date = st.date_input("Selecciona fecha", value=max_date, min_value=min_date, max_value=max_date)
                chosen_rows = df_lottery[df_lottery['Fecha'].dt.date == selected_date]
                if chosen_rows.empty:
                    st.info("No hay resultado exacto para esa fecha. Mostrando el último disponible antes de la fecha seleccionada.")
                    chosen_rows = df_lottery[df_lottery['Fecha'].dt.date <= selected_date].tail(1)
                st.subheader("Resultado")
                st.table(chosen_rows.reset_index(drop=True))

    # --- Prediction tab ---
    with tabs[1]:
        st.header("Predicción automática (sin inputs manuales)")
        st.write("La predicción trabaja con la última secuencia histórica y determina el siguiente sorteo objetivo.")
        if df is None or df.empty:
            st.warning("No hay historial para predecir. Ejecuta el actualizador o sube el CSV.")
        elif keras_model is None:
            st.warning("No se detectó modelo Keras en la raíz (bolitero_nivel_dios_model.h5). Sin predicción.")
        else:
            st.write(f"Historial disponible: {len(df)} filas. Último registro: {df['Fecha'].max()}")
            if st.button("Generar predicción automática"):
                try:
                    latest_row = df.iloc[-1]
                    ultimo_estado = latest_row['Estado']
                    ultimo_horario = latest_row['Horario']
                    ultima_fecha = latest_row['Fecha']
                    proximo_estado, proximo_horario = obtener_proximo_sorteo(ultimo_estado, ultimo_horario)
                    X_input = prepare_latest_sequence(df, SEQUENCE_LENGTH, scaler, encoder)
                    plan = generar_jugada_estrategica_pro(keras_model, X_input)
                    plan['Sorteo objetivo'] = f"{proximo_estado} {proximo_horario} del {ultima_fecha.date()}"
                    st.success("Predicción generada")
                    st.json(plan)
                except Exception as e:
                    st.error(f"Error generando predicción: {e}")

    # --- Historial tab ---
    with tabs[2]:
        st.header("Historial completo")
        if df is None or df.empty:
            st.warning("No hay CSV histórico cargado.")
            uploaded = st.file_uploader("Sube un CSV de historiales (temporal)", type=["csv"])
            if uploaded:
                udf = pd.read_csv(uploaded)
                st.dataframe(udf.head(200))
        else:
            st.dataframe(df)
            st.download_button("Descargar CSV", df.to_csv(index=False), file_name="historico_bolita_MASTER_export.csv")

    # Sidebar quick info
    st.sidebar.markdown("---")
    st.sidebar.write({
        "CSV existe": csv_path.exists(),
        "Modelo Keras": model_h5.exists(),
        "Scaler": scaler_pkl.exists(),
        "Encoder": encoder_pkl.exists(),
        "Filas historial": len(df) if df is not None else 0,
    })

if __name__ == '__main__':
    main()
