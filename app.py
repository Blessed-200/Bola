import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------------------
# PARTE 1: DEFINICIÓN DE LAS FUNCIONES (Mover funciones auxiliares)
# ----------------------------------------------------------------------------------

# 1. FUNCIÓN DE CARGA CRÍTICA CON CACHÉ
# Usar st.cache_resource para cargar el modelo y transformadores una sola vez.
# Esto es CRÍTICO, ya que load_model es una operación costosa.
@st.cache_resource
def load_critical_files():
    try:
        bolitero_model = load_model('bolitero_nivel_dios_model.h5')
        scaler = joblib.load('minmax_scaler.pkl')
        encoder = joblib.load('onehot_encoder.pkl')
        # NOTA: Cargar el CSV directamente en la función principal para permitir su actualización
        return bolitero_model, scaler, encoder
    except Exception as e:
        st.error(f"❌ ERROR: Fallo al cargar archivos críticos (modelo/transformadores). Asegúrate de que estén en la misma carpeta. Error: {e}")
        return None, None, None

# 2. FUNCIÓN DE SECUENCIACIÓN (Igual que antes)
def prepare_latest_sequence(df_master, sequence_length, scaler_num, encoder_cat):
    # ... (El código de tu función 'prepare_latest_sequence' va aquí, sin cambios)
    df_sequence = df_master.tail(sequence_length).copy()
    numerical_features = ['Centena', 'Fijo', 'Corrido_1', 'Corrido_2']
    categorical_features = ['Estado', 'Horario']
    
    df_sequence[numerical_features] = scaler_num.transform(df_sequence[numerical_features])
    encoded_features = encoder_cat.transform(df_sequence[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder_cat.get_feature_names_out(categorical_features))
    
    df_processed = df_sequence.drop(categorical_features + ['Fecha'], axis=1).reset_index(drop=True)
    # Aquí puedes necesitar hacer un reset_index() en encoded_df si el encoder devuelve un array de numpy
    # Vamos a asumir que el encoder devuelve algo compatible para concatenar:
    df_processed = pd.concat([df_processed, encoded_df], axis=1) 
    
    # Asegurarse de que el orden de las columnas sea el esperado por el modelo
    # El modelo espera todas las columnas excepto 'Fijo'. Vamos a extraer el orden del encoder si es posible
    # Si da error en la línea anterior, revisa el formato de 'encoded_df'
    column_order = [col for col in df_processed.columns if col != 'Fijo']
    X_sequence = df_processed[column_order].values # Asegurar orden de las columnas
    X_input = X_sequence.reshape(1, sequence_length, X_sequence.shape[1])
    
    return X_input

# 3. FUNCIÓN DE DEDUCCIÓN DEL SORTEO OBJETIVO (Igual que antes)
def obtener_proximo_sorteo(ultimo_estado, ultimo_horario):
    # ... (El código de tu función 'obtener_proximo_sorteo' va aquí, sin cambios)
    secuencia_de_juego = {
        ('GA', 'Night'): ('TN', 'Morning'),
        ('TN', 'Morning'): ('GA', 'Midday'),
        ('GA', 'Midday'): ('NJ', 'Midday'),
        ('NJ', 'Midday'): ('FL', 'Midday'),
        ('FL', 'Midday'): ('NY', 'Midday'),
        ('NY', 'Midday'): ('GA', 'Evening'),
        ('GA', 'Evening'): ('FL', 'Evening'),
        ('FL', 'Evening'): ('NY', 'Evening'),
        ('NY', 'Evening'): ('NJ', 'Evening'),
        ('NJ', 'Evening'): ('GA', 'Night'),
    }
    return secuencia_de_juego.get((ultimo_estado, ultimo_horario), ('ERROR', 'REVISAR'))

# 4. FUNCIÓN DE JUGADA ESTRATÉGICA PRO (Igual que antes)
def generar_jugada_estrategica_pro(model, latest_sequence, prob_objetivo=0.80, max_numeros=20):
    # ... (El código de tu función 'generar_jugada_estrategica_pro' va aquí, sin cambios)
    predictions = model.predict(latest_sequence, verbose=0) # Añadir verbose=0 para Streamlit
    fijo_probs = predictions[0][0] 
    fijo_ranking_indices = np.argsort(fijo_probs)[::-1] 
    
    jugada_fijo_duro = []
    probabilidad_acumulada = 0.0
    
    for idx in fijo_ranking_indices:
        prob = fijo_probs[idx]
        jugada_fijo_duro.append(idx)
        probabilidad_acumulada += prob

        if probabilidad_acumulada >= prob_objetivo and len(jugada_fijo_duro) >= 10:
             break
    
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
        "🎯 Sorteo a Predecir": "Placeholder", 
        "🎯 Probabilidad Acumulada (ROI):": f"{min(probabilidad_acumulada * 100, 100):.2f}%",
        "🛑 Fijos (Duro - Top 10):": [f"{n:02d}" for n in jugada_fijo_duro[:10]], 
        "🟢 Fijos (Cobertura):": [f"{n:02d}" for n in jugada_fijo_duro[10:]], 
        "Total de Números Únicos:": len(jugada_fijo_duro),
        "⭐ Estrategia de Parlet (Dos de Tres):": "\n".join(parlet_plan),
        "💡 Cobertura Extra (Jugada de Ventaja):": f"Jugar T{top_terminal} y D{top_decena} por separado."
    }
    
    return plan_de_juego

# ----------------------------------------------------------------------------------
# PARTE 2: FUNCIÓN PRINCIPAL DE STREAMLIT
# ----------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Bolitero Nivel Dios", layout="wide")
    st.title("🔥 PLAN DE JUEGO BOLITERO NIVEL DIOS 🔥")
    st.markdown("---")
    
    SEQUENCE_LENGTH = 7 

    # 1. CARGAR ARCHIVOS CRÍTICOS (Usando la función de caché)
    bolitero_model, scaler, encoder = load_critical_files()

    if bolitero_model is None:
        st.stop() # Detener la ejecución si falla la carga del modelo/transformadores

    # 2. CARGAR Y PREPARAR DATOS (Usar @st.cache_data para el CSV si es grande)
    # Por ahora, se carga en cada ejecución, pero si el CSV no cambia a menudo, usa @st.cache_data
    try:
        df_actualizado = pd.read_csv('historico_bolita_MASTER.csv')
        df_actualizado = df_actualizado.sort_values(by=['Fecha', 'Horario'], ascending=True)
        # Mostrar el último sorteo procesado
        ultima_fila = df_actualizado.iloc[-1] 
        ultimo_estado = ultima_fila['Estado']
        ultimo_horario = ultima_fila['Horario']
        ultima_fecha = ultima_fila['Fecha']
        
        st.info(f"✅ Datos cargados. Último Sorteo Procesado: **{ultimo_estado} {ultimo_horario}** del **{ultima_fecha}**.")
    except Exception as e:
        st.error(f"❌ ERROR: Fallo al cargar el archivo de datos 'historico_bolita_MASTER.csv'. Error: {e}")
        return

    # 3. BOTÓN DE PREDICCIÓN
    if st.button("🔮 Generar Jugada Estratégica", type="primary"):
        with st.spinner('Analizando secuencia y generando la predicción...'):
            # IDENTIFICAR EL SORTEO OBJETIVO
            proximo_estado, proximo_horario = obtener_proximo_sorteo(ultimo_estado, ultimo_horario)
            
            # GENERAR LA PREDICCIÓN
            latest_sequence_input = prepare_latest_sequence(df_actualizado, SEQUENCE_LENGTH, scaler, encoder) 
            plan_de_jugada = generar_jugada_estrategica_pro(bolitero_model, latest_sequence_input)

            # ACTUALIZAR EL REPORTE CON EL SORTEO OBJETIVO
            plan_de_jugada['🎯 Sorteo a Predecir'] = f"**{proximo_estado} {proximo_horario}** del **{ultima_fecha}** (Día siguiente o siguiente horario)"

            # 4. MOSTRAR EL PLAN FINAL EN PANELES
            st.success(f"🎉 **¡PLAN GENERADO!** Esta predicción es para: {plan_de_jugada['🎯 Sorteo a Predecir']}")
            st.markdown("---")

            col1, col2 = st.columns([1, 1])

            # Columna 1: Fijos y ROI
            with col1:
                st.subheader("🎯 Fijos y Potencial de Retorno (ROI)")
                st.metric(label="Probabilidad Acumulada (ROI)", 
                          value=plan_de_jugada["🎯 Probabilidad Acumulada (ROI):"])
                st.write(f"Total de Números Únicos: **{plan_de_jugada['Total de Números Únicos:']}**")
                
                st.markdown("### 🛑 Fijos (Juego Duro - Top 10)")
                # Muestra los números en un formato de grilla para mejor visualización
                fijos_duros = plan_de_jugada['🛑 Fijos (Duro - Top 10):']
                st.code(", ".join(fijos_duros))

                if plan_de_jugada['🟢 Fijos (Cobertura):']:
                    st.markdown("### 🟢 Fijos (Cobertura Extra)")
                    fijos_cobertura = plan_de_jugada['🟢 Fijos (Cobertura):']
                    st.code(", ".join(fijos_cobertura))

            # Columna 2: Estrategia de Parlet y Cobertura
            with col2:
                st.subheader("⭐ Estrategia de Parlet y Cobertura")
                st.markdown("### Estrategia de Parlet (Dos de Tres)")
                # Usar st.markdown con un listado o st.text
                for parlet in plan_de_jugada["⭐ Estrategia de Parlet (Dos de Tres):"].split('\n'):
                    st.write(f"- {parlet}")
                
                st.markdown("### 💡 Cobertura de Ventaja")
                st.info(plan_de_jugada["💡 Cobertura Extra (Jugada de Ventaja):"])
            
            st.markdown("---")
            st.dataframe(df_actualizado.tail(SEQUENCE_LENGTH + 1)) # Mostrar la secuencia usada para la predicción

if __name__ == '__main__':
    main()
