# --- SCRIPT DE STREAMLIT: BOLITERO MODO DIOS PRO ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os
from tensorflow.keras.models import load_model

# Configuración de la Página
st.set_page_config(
    page_title="Bolitero Modo Dios Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variables de Configuración Globales (Los sorteos que nos interesan)
TARGET_DRAWS = [('GA', 'Midday'), ('FL', 'Midday'), ('NY', 'Midday'), 
                ('GA', 'Evening'), ('FL', 'Evening'), ('NY', 'Evening'), 
                ('GA', 'Night')]
SEQUENCE_LENGTH = 20
categorical_features = ['Estado', 'Horario']

# ----------------------------------------------------------------------------------
# PARTE 1: FUNCIONES ESENCIALES Y LÓGICA DE DATOS
# ----------------------------------------------------------------------------------

# Carga de Datos y Herramientas (Cacheadas para rendimiento en Streamlit)
@st.cache_resource
def load_assets():
    """Carga todos los modelos y preprocesadores una sola vez."""
    try:
        # Cargar los 7 modelos. Aquí usamos el modelo único por ahora para no fallar.
        # ADVERTENCIA: Debes reemplazar esto con la lógica de 7 modelos individuales después.
        model = load_model('bolitero_nivel_dios_model.h5')
        scaler = joblib.load('minmax_scaler.pkl') # Asumiendo este es el feature_only_scaler
        encoder = joblib.load('onehot_encoder.pkl')
        
        # Cargar el CSV master
        df = pd.read_csv('historico_bolita_MASTER.csv')
        df = df.sort_values(by=['Fecha', 'Horario'], ascending=True).reset_index(drop=True)
        return model, scaler, encoder, df
    except Exception as e:
        st.error(f"❌ ERROR CRÍTICO al cargar archivos: {e}")
        st.stop()
        return None, None, None, None

# 1. FUNCIÓN DE SECUENCIACIÓN (Prepara los últimos 7 sorteos)
def prepare_latest_sequence(df_master, sequence_length, scaler_num, encoder_cat):
    # Lógica de preprocesamiento (mantenida del código original)
    df_sequence = df_master.tail(sequence_length).copy()
    # ... (cuerpo de la función prepare_latest_sequence)
    numerical_features = ['Centena', 'Fijo', 'Corrido_1', 'Corrido_2']
    
    df_sequence[numerical_features] = scaler_num.transform(df_sequence[numerical_features])
    encoded_features = encoder_cat.transform(df_sequence[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder_cat.get_feature_names_out(categorical_features))
    
    df_processed = df_sequence.drop(categorical_features + ['Fecha'], axis=1).reset_index(drop=True)
    df_processed = pd.concat([df_processed, encoded_df], axis=1)
    
    X_sequence = df_processed.drop(['Fijo'], axis=1).values
    X_input = X_sequence.reshape(1, sequence_length, X_sequence.shape[1])
    return X_input

# 2. FUNCIÓN DE DEDUCCIÓN DEL SORTEO OBJETIVO (Tu Lógica de Secuencia)
def obtener_proximo_sorteo(ultimo_estado, ultimo_horario):
    # ... (cuerpo de la función obtener_proximo_sorteo)
    secuencia_de_juego = {
        ('GA', 'Night'): ('TN', 'Morning'), ('TN', 'Morning'): ('GA', 'Midday'),
        ('GA', 'Midday'): ('NJ', 'Midday'), ('NJ', 'Midday'): ('FL', 'Midday'),
        ('FL', 'Midday'): ('NY', 'Midday'), ('NY', 'Midday'): ('GA', 'Evening'),
        ('GA', 'Evening'): ('FL', 'Evening'), ('FL', 'Evening'): ('NY', 'Evening'),
        ('NY', 'Evening'): ('NJ', 'Evening'), ('NJ', 'Evening'): ('GA', 'Night'),
    }
    
    estado, horario = secuencia_de_juego.get((ultimo_estado, ultimo_horario), ('ERROR', 'REVISAR'))
    
    # Salta sorteos que no juegas (TN, NJ), asegurando la continuidad
    while estado in ['TN', 'NJ']:
        ultimo_estado, ultimo_horario = estado, horario
        estado, horario = secuencia_de_juego.get((ultimo_estado, ultimo_horario), ('ERROR', 'REVISAR'))
        
    return estado, horario

# 3. FUNCIÓN DE JUGADA ESTRATÉGICA PRO (ROI + Parlet)
def generar_jugada_estrategica_pro(model, latest_sequence, prob_objetivo=0.80):
    # ... (cuerpo de la función generar_jugada_estrategica_pro - sin cambios)
    predictions = model.predict(latest_sequence)
    fijo_probs = predictions[0][0] 
    fijo_ranking_indices = np.argsort(fijo_probs)[::-1] 
    
    jugada_fijo_duro = []
    probabilidad_acumulada = 0.0
    
    for idx in fijo_ranking_indices:
        prob = fijo_probs[idx]
        jugada_fijo_duro.append(idx)
        probabilidad_acumulada += prob

        # Si ya alcanzó el 80% y tiene al menos 10 números, detente.
        if probabilidad_acumulada >= prob_objetivo and len(jugada_fijo_duro) >= 10:
             break
    
    # Asume modelo de múltiples salidas (Tal como está en tu archivo original)
    top_terminal = np.argmax(predictions[1][0])
    top_decena = np.argmax(predictions[2][0])

    top_fijos_parlet = sorted(jugada_fijo_duro[:5], key=lambda x: fijo_probs[x], reverse=True) 

    parlet_plan = []
    for i in range(len(top_fijos_parlet)):
        for j in range(i + 1, len(top_fijos_parlet)):
            parlet_plan.append(f"{top_fijos_parlet[i]:02d} x {top_fijos_parlet[j]:02d}")
    
    if top_fijos_parlet:
        parlet_plan.append(f"COBERTURA DURA: {top_fijos_parlet[0]:02d} x D{top_decena} o T{top_terminal}")
    
    plan_de_juego = {
        "🎯 Probabilidad Acumulada (ROI):": f"{min(probabilidad_acumulada * 100, 100):.2f}%",
        "🛑 Fijos (Duro - Top 10):": [f"{n:02d}" for n in jugada_fijo_duro[:10]], 
        "🟢 Fijos (Cobertura):": [f"{n:02d}" for n in jugada_fijo_duro[10:]], 
        "Total de Números Únicos:": len(jugada_fijo_duro),
        "⭐ Estrategia de Parlet (Dos de Tres):": "\n".join(parlet_plan),
        "💡 Cobertura Extra (Jugada de Ventaja):": f"Jugar T{top_terminal} y D{top_decena} por separado."
    }
    
    return plan_de_juego

# ----------------------------------------------------------------------------------
# PARTE 2: LÓGICA DEL STREAMLIT (UI y Flujo de Navegación)
# ----------------------------------------------------------------------------------

# Cargar activos
bolitero_model, scaler, encoder, df_master = load_assets()

st.title("🔥 Bolitero Maestro: Modo Dios Pro")

# Usar tabs para la navegación
tab_dashboard, tab_prediccion = st.tabs(["1. Dashboard Histórico y Actualización", "2. Predicción Ojo Clínico"])

# --- TABLA 1: DASHBOARD HISTÓRICO Y ACTUALIZACIÓN ---
with tab_dashboard:
    st.header("Gestión y Exploración de Datos Históricos")

    # Lógica de Actualización de Datos (Hard Refresh)
    st.subheader("Actualización de la Base de Datos")
    st.markdown("""
        ⚠️ Para actualizar, tu *scraper* (`.py`) debe ejecutarse y subir la versión más reciente 
        del `historico_bolita_MASTER.csv` a tu repositorio de GitHub.
        """)
    
    if st.button("Actualizar y Recargar Base de Datos desde GitHub"):
        # Forzar a Streamlit a limpiar su caché y recargar los datos
        st.cache_resource.clear() # La forma moderna de limpiar el caché
        st.rerun() # LA FUNCIÓN CORREGIDA
        st.toast("✅ Base de datos recargada. Confirma el cambio de fecha máxima.")


    st.divider()

    # Filtro de Fechas (Para manipular el CSV)
    st.subheader("Visualización del Histórico (GA, FL, NY)")
    
    df_filtered_draws = df_master[df_master['Estado'].isin(['GA', 'FL', 'NY'])].copy()
    
    min_date = pd.to_datetime(df_filtered_draws['Fecha']).min().date()
    max_date = pd.to_datetime(df_filtered_draws['Fecha']).max().date()

    selected_date = st.date_input(
        "Selecciona la Fecha a Visualizar:",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Mostrar la tabla filtrada
    st.dataframe(
        df_filtered_draws[df_filtered_draws['Fecha'] == str(selected_date)],
        use_container_width=True
    )

# --- TABLA 2: PREDICCIÓN OJO CLÍNICO ---
with tab_prediccion:
    st.header("Generar Plan de Jugada de Máxima Certeza")

    # 1. IDENTIFICAR SORTEO OBJETIVO (Lógica corregida)
    
    # Encontrar la última fila donde el Estado sea GA, FL o NY
    df_juego = df_master[df_master['Estado'].isin(['GA', 'FL', 'NY'])].copy()
    
    if df_juego.empty:
        st.error("No hay sorteos de GA, FL o NY para iniciar la secuencia.")
        st.stop()
        
    ultima_fila_relevante = df_juego.iloc[-1] 
    
    ultimo_estado = ultima_fila_relevante['Estado']
    ultimo_horario = ultima_fila_relevante['Horario']
    ultima_fecha = ultima_fila_relevante['Fecha']
    
    proximo_estado, proximo_horario = obtener_proximo_sorteo(ultimo_estado, ultimo_horario)

    st.info(f"""
        **ÚLTIMO SORTEO BASE CONCLUIDO:** {ultimo_estado} {ultimo_horario} del {ultima_fecha} 
        (Este resultado dispara la secuencia de predicción).
    """)
    st.markdown(f"## 🎯 **¡PREDICCIÓN OBJETIVO:** **{proximo_estado} {proximo_horario}**")
    
    st.divider()

    # 2. GENERAR LA PREDICCIÓN AL PRESIONAR EL BOTÓN
    if st.button("🚀 Generar Plan de Jugada MODO DIOS", type="primary"):
        
        with st.spinner('Analizando secuencia histórica y activando Ojo Clínico...'):
            try:
                # Obtener la secuencia de 7 (incluyendo sorteos TN/NJ que son el contexto)
                latest_sequence_input = prepare_latest_sequence(df_master, SEQUENCE_LENGTH, scaler, encoder) 
                
                # Generar el plan
                plan_de_jugada = generar_jugada_estrategica_pro(bolitero_model, latest_sequence_input)

                st.success("✅ Predicción y Plan de Jugada generados con éxito.")

                # 3. MOSTRAR EL PLAN FINAL (Pendiente de tu formato final)
                st.subheader("Plan de Juego Sugerido")
                
                col_roi, col_total = st.columns(2)
                col_roi.metric(
                    "Probabilidad Acumulada (ROI)",
                    plan_de_jugada['🎯 Probabilidad Acumulada (ROI):']
                )
                col_total.metric(
                    "Total de Números Únicos",
                    plan_de_jugada['Total de Números Únicos:']
                )
                
                # Mostrar Fijos y Cobertura
                col_fijos, col_parlet = st.columns(2)
                with col_fijos:
                    st.markdown("#### Fijos a Jugar (Cobertura Base)")
                    st.code(", ".join(plan_de_jugada['🛑 Fijos (Duro - Top 10):']))
                    
                with col_parlet:
                    st.markdown("#### Plan de Parlet y Cobertura Dura")
                    st.code(plan_de_jugada['⭐ Estrategia de Parlet (Dos de Tres):'])
                    st.info(f"Cobertura Extra: {plan_de_jugada['💡 Cobertura Extra (Jugada de Ventaja):']}")

                with st.expander("Ver Fijos de Cobertura Extra"):
                    st.text(", ".join(plan_de_jugada['🟢 Fijos (Cobertura):']))
                    
            except Exception as e:
                st.error(f"❌ ERROR durante la predicción. Detalle: {e}")

