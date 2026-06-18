import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Page Config ---
st.set_page_config(
    page_title="CardioCheck AI - Diagnostic Support",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Google Font & Premium CSS Injection ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Main font styling */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphic card styling */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
    margin-bottom: 25px;
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
}

/* Neon title text */
.header-title {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.2rem;
    margin-bottom: 5px;
}

.sub-title {
    color: rgba(255, 255, 255, 0.6);
    font-size: 1.1rem;
    margin-bottom: 30px;
}

/* Prediction Alert Styling */
.alert-card-success {
    background: rgba(0, 255, 135, 0.08);
    border: 1px solid rgba(0, 255, 135, 0.25);
    border-radius: 16px;
    padding: 20px;
    margin-top: 15px;
    box-shadow: 0 4px 20px rgba(0, 255, 135, 0.1);
}

.alert-card-danger {
    background: rgba(255, 8, 68, 0.08);
    border: 1px solid rgba(255, 8, 68, 0.25);
    border-radius: 16px;
    padding: 20px;
    margin-top: 15px;
    box-shadow: 0 4px 20px rgba(255, 8, 68, 0.1);
}

/* Form Styling tweaks */
.stButton>button {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 25px !important;
    transition: box-shadow 0.3s ease, transform 0.3s ease !important;
}

.stButton>button:hover {
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.6) !important;
    transform: scale(1.02) !important;
}
</style>
""", unsafe_allow_html=True)

# --- Load Model & Preprocessor ---
@st.cache_resource
def load_assets():
    model_path = "app/deployment_model.pkl"
    prep_path = "app/deployment_preprocessor.pkl"
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        return None, None
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    return model, preprocessor

model, preprocessor = load_assets()

# --- Helper function for SVG risk gauge ---
def get_risk_gauge(probability):
    percentage = int(probability * 100)
    
    # Choose color scheme based on risk
    if probability < 0.3:
        color = "#00ff87" # Green
        label = "RISQUE FAIBLE"
    elif probability < 0.6:
        color = "#f9d423" # Yellow
        label = "RISQUE MODÉRÉ"
    else:
        color = "#ff0844" # Red
        label = "RISQUE ÉLEVÉ"
        
    stroke_dashoffset = 314.16 - (314.16 * probability)
    
    svg_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 10px 0;">
        <svg width="180" height="180" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255, 255, 255, 0.05)" stroke-width="8" />
            <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="8" 
                    stroke-dasharray="314.16" stroke-dashoffset="{stroke_dashoffset}" 
                    stroke-linecap="round" transform="rotate(-90 60 60)" 
                    style="transition: stroke-dashoffset 1s ease-out; filter: drop-shadow(0px 0px 8px {color});" />
            <text x="60" y="62" text-anchor="middle" fill="#ffffff" font-size="20" font-weight="800" font-family="'Outfit', sans-serif">
                {percentage}%
            </text>
            <text x="60" y="78" text-anchor="middle" fill="{color}" font-size="7" font-weight="700" font-family="'Outfit', sans-serif" letter-spacing="1">
                {label}
            </text>
        </svg>
    </div>
    """
    return svg_html

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🫀 CardioCheck AI")
    st.markdown("Système de support à la décision clinique pour la prédiction du risque coronaire.")
    st.markdown("---")
    
    if model is not None:
        st.success("✅ Modèle chargé avec succès")
        st.info(f"Algorithme : **{type(model).__name__}**")
    else:
        st.error("❌ Modèle non trouvé")
        st.warning("Veuillez d'abord exécuter `train_local.py` pour entraîner et exporter le modèle.")
        
    st.markdown("---")
    st.markdown("#### Métriques cliniques suivies")
    st.markdown("""
    - Pression artérielle (systolique)
    - Taux de cholestérol sérique
    - Dépression du segment ST (oldpeak)
    - Fréquence cardiaque maximale (thalach)
    - Nombre de vaisseaux majeurs colorés (ca)
    """)
    st.markdown("---")
    st.caption("Avis de non-responsabilité : Cette application est un prototype à visée académique et ne remplace en aucun cas un diagnostic médical professionnel.")

# --- Main Layout ---
st.markdown('<div class="header-title">🫀 CardioCheck AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligence Artificielle pour la Prédiction du Risque Cardiaque</div>', unsafe_allow_html=True)

if model is None or preprocessor is None:
    st.error("### Modèle ou préprocesseur introuvable !")
    st.markdown("""
    Pour générer le modèle et faire fonctionner cette application :
    1. Ouvrez un terminal dans le dossier `LouaML`.
    2. Exécutez le script d'entraînement :
    ```bash
    python3 train_local.py
    ```
    3. Actualisez cette page.
    """)
else:
    # --- Input Form ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Formulaire d'évaluation du patient")
    
    tab1, tab2 = st.tabs(["📋 Données Cliniques", "🩺 Examens & Diagnostics"])
    
    with st.form("patient_assessment_form"):
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Âge du patient", min_value=20, max_value=100, value=55, help="Âge en années.")
                sex = st.selectbox("Sexe", ["Femme", "Homme"], help="Sexe biologique.")
                cp = st.selectbox(
                    "Type de douleur thoracique (cp)",
                    [
                        ("Angine typique (1)", 1),
                        ("Angine atypique (2)", 2),
                        ("Douleur non angineuse (3)", 3),
                        ("Asymptomatique (4)", 4)
                    ],
                    format_func=lambda x: x[0],
                    help="Description de la douleur thoracique ressentie."
                )
            with col2:
                trestbps = st.number_input("Pression artérielle au repos (trestbps)", min_value=80, max_value=200, value=120, help="Pression artérielle systolique en mm Hg à l'admission.")
                chol = st.number_input("Cholestérol sérique (chol)", min_value=100, max_value=600, value=220, help="Taux de cholestérol sérique en mg/dl.")
                fbs = st.selectbox("Glycémie à jeun > 120 mg/dl (fbs)", ["Non", "Oui"], help="Taux de sucre dans le sang à jeun.")
                
        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                restecg = st.selectbox(
                    "Résultats ECG au repos (restecg)",
                    [
                        ("Normal (0)", 0),
                        ("Anomalie de la vague ST-T (1)", 1),
                        ("Hypertrophie ventriculaire gauche (2)", 2)
                    ],
                    format_func=lambda x: x[0],
                    help="Résultats de l'électrocardiogramme au repos."
                )
                thalach = st.number_input("Fréquence cardiaque max (thalach)", min_value=70, max_value=220, value=150, help="Fréquence cardiaque maximale atteinte lors de l'effort.")
                exang = st.selectbox("Angine induite par l'effort (exang)", ["Non", "Oui"], help="Présence d'une angine de poitrine déclenchée par l'exercice.")
                oldpeak = st.number_input("Dépression ST (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, help="Dépression du segment ST induite par l'effort par rapport au repos.")
            with col4:
                slope = st.selectbox(
                    "Pente du segment ST à l'effort (slope)",
                    [
                        ("Montante (1)", 1),
                        ("Plate (2)", 2),
                        ("Descendante (3)", 3)
                    ],
                    format_func=lambda x: x[0],
                    help="La pente du segment ST d'effort maximal."
                )
                ca = st.selectbox("Nombre de vaisseaux majeurs colorés (ca)", [0, 1, 2, 3], help="Nombre de vaisseaux majeurs (0-3) colorés par fluoroscopie.")
                thal = st.selectbox(
                    "Thalassémie (thal)",
                    [
                        ("Normal (3)", 3),
                        ("Défaut réversible (7)", 7),
                        ("Défaut fixé (6)", 6)
                    ],
                    format_func=lambda x: x[0],
                    help="Type de thalassémie détecté."
                )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🩺 Lancer l'analyse diagnostique")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Prediction Output ---
    if submitted:
        # Map input to model schema
        sex_val = 1 if sex == "Homme" else 0
        cp_val = cp[1]
        fbs_val = 1 if fbs == "Oui" else 0
        restecg_val = restecg[1]
        exang_val = 1 if exang == "Oui" else 0
        slope_val = slope[1]
        thal_val = thal[1]
        
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex_val,
            'cp': cp_val,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs_val,
            'restecg': restecg_val,
            'thalach': thalach,
            'exang': exang_val,
            'oldpeak': oldpeak,
            'slope': slope_val,
            'ca': ca,
            'thal': thal_val
        }])
        
        # Preprocess & Predict
        try:
            X_proc = preprocessor.transform(input_data)
            proba = model.predict_proba(X_proc)[0, 1]
            pred = model.predict(X_proc)[0]
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 Résultats de l'évaluation")
            
            col_gauge, col_details = st.columns([1, 1.2])
            
            with col_gauge:
                st.markdown(get_risk_gauge(proba), unsafe_allow_html=True)
                
            with col_details:
                if pred == 1:
                    st.markdown("""
                    <div class="alert-card-danger">
                        <h4 style="color: #ff0844; margin: 0 0 5px 0;">Risque Élevé Détecté</h4>
                        <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
                            Le modèle indique une forte probabilité de cardiopathie. Une consultation médicale urgente est recommandée.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-card-success">
                        <h4 style="color: #00ff87; margin: 0 0 5px 0;">Risque Faible Détecté</h4>
                        <p style="margin: 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
                            Les indicateurs physiologiques du patient se situent dans des plages de risque bas à modéré.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Clinical Details Table
                st.markdown("<br>", unsafe_allow_html=True)
                details_df = pd.DataFrame({
                    "Paramètre": ["Probabilité", "Âge", "Tension Repos", "Cholestérol", "Fréquence Max"],
                    "Valeur": [f"{proba:.2%}", f"{age} ans", f"{trestbps} mm Hg", f"{chol} mg/dl", f"{thalach} bpm"]
                })
                st.dataframe(details_df, use_container_width=True, hide_index=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # --- Report Generator ---
            report_text = f"""==================================================
CARDIOCHECK AI - PATIENT HEALTH ASSESSMENT REPORT
==================================================
Age: {age}
Sex: {sex}
Chest Pain Type: {cp[0]}
Resting Blood Pressure: {trestbps} mm Hg
Serum Cholesterol: {chol} mg/dl
Fasting Blood Sugar > 120 mg/dl: {fbs}
Resting ECG: {restecg[0]}
Max Heart Rate Achieved: {thalach} bpm
Exercise Induced Angina: {exang}
ST Depression (oldpeak): {oldpeak}
ST Slope: {slope[0]}
Major Vessels Colored (ca): {ca}
Thalassemia Type: {thal[0]}
--------------------------------------------------
DIAGNOSTIC PREDICTION
--------------------------------------------------
Prediction Class: {"Heart Disease Detected (High Risk)" if pred == 1 else "Normal (Low Risk)"}
Risk Probability: {proba:.4f} ({proba:.2%})
--------------------------------------------------
Clinical Guidance:
{ "The model highlights a significantly elevated risk of heart disease. Consult a cardiologist for comprehensive diagnostic testing." if pred == 1 else "Indicators remain normal. Encourage general healthy lifestyle guidelines and routine medical checks." }
=================================================="""
            
            st.download_button(
                label="📥 Télécharger le rapport clinique (TXT)",
                data=report_text,
                file_name=f"rapport_cardio_{age}_{sex_val}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Erreur de prédiction : {str(e)}")
            st.info("Vérifiez que le modèle et le préprocesseur correspondent aux données du formulaire.")
