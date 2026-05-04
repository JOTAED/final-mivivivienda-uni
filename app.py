import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de página
st.set_page_config(page_title="Analizador MIVIVIENDA", page_icon="🏦", layout="centered")

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_final.pkl")

model = load_model()

st.title("🏦 Clasificador de Entidades Financieras")
st.markdown("### Edwin Mejia Angeles - PC2 Ciencia de Datos")
st.write("Ingrese los datos del crédito para predecir si corresponde a **Banca** o **Microfinanzas**.")

# --- FORMULARIO DE ENTRADA ---
with st.form("pred_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        producto = st.selectbox("Producto", ["CMV", "CTP", "MC", "CTP_P", "CMV_P"])
        monto_vivienda = st.number_input("Valor de Vivienda (S/.)", value=150000)
        monto_credito = st.number_input("Monto del Crédito (S/.)", value=120000)
        tasa = st.number_input("Tasa de Interés (%)", 1.0, 40.0, 10.5, 0.01)
        
    with col2:
        departamento = st.selectbox("Departamento", ["LIMA", "ICA", "JUNIN", "PIURA", "AREQUIPA", "LA LIBERTAD", "LAMBAYEQUE", "OTROS"])
        plazos = st.select_slider("Plazo (Meses)", options=[60, 120, 180, 240, 300])
        bono = st.selectbox("Grado Bono Sostenible", [0, 1, 2, 3])
        bbp = st.number_input("Monto BBP (S/.)", value=15000)

    submit = st.form_submit_button("🚀 Realizar Predicción")

# --- LÓGICA DE PREDICCIÓN ---
if submit:
    # Crear el DataFrame con el mismo nombre de columnas que el entrenamiento
    input_df = pd.DataFrame([{
        "PRODUCTO": producto,
        "MONTO_CREDITO": monto_credito,
        "MONTO_CUOTA_INICIAL": monto_vivienda - monto_credito,
        "PLAZOS": plazos,
        "TASA": tasa,
        "MONTO_VALOR_VIVIENDA": monto_vivienda,
        "DEPARTAMENTO": departamento,
        "GRADO_BONO_SOSTENIBLE": bono,
        "MONTO_BBP": bbp
    }])

    # Predicción
    prediccion = model.predict(input_df)[0]
    probabilidades = model.predict_proba(input_df)
    confianza = np.max(probabilidades) * 100

    # Mostrar Resultados
    st.divider()
    if prediccion == "Banca":
        st.success(f"### Resultado: **{prediccion}**")
    else:
        st.info(f"### Resultado: **{prediccion}**")
        
    st.write(f"**Nivel de confianza:** {confianza:.2f}%")
    
    # Mensaje interpretativo
    if tasa > 15:
        st.warning("Nota: La tasa elevada es un factor determinante para Microfinanzas.")
    elif monto_credito > 200000:
        st.success("Nota: El monto alto es un factor determinante para Banca.")