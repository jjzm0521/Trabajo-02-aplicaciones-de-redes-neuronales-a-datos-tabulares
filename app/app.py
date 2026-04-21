"""
Score de Riesgo Crediticio — Aplicacion Web
============================================
Variables visibles (4) — accesibles directamente por el cliente:
    loan_amnt, term, annual_inc, tot_cur_bal

Variables fijas a la mediana/moda (5) — no conocidas por el cliente
o con ambigüedad temporal en el dataset:
    dti                 → cargada desde outputs/defaults.json
    revol_util          → cargada desde outputs/defaults.json
    installment         → cargada desde outputs/defaults.json (leakage documentado)
    total_rev_hi_lim    → cargada desde outputs/defaults.json
    verification_status → cargada desde outputs/defaults.json

Nota sobre el pipeline:
    La Red Neuronal fue entrenada directamente sobre los valores WOE sin
    normalización previa (ver Notebook 03, clase RedNeuronal). El scaler
    solo se utiliza aquí para recuperar el orden canónico de las features.

Ejecutar con:  streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import json
import joblib
import torch
import torch.nn as nn
import scorecardpy as sc
import warnings

# ─── Configuracion ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditScore — Tu Riesgo Crediticio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Arquitectura de la Red Neuronal ─────────────────────────────────────────
class RedNeuronal(nn.Module):
    def __init__(self, n_entrada, capas, dropout):
        super(RedNeuronal, self).__init__()
        bloques = []
        n_anterior = n_entrada
        for n_neuronas in capas:
            bloques.append(nn.Linear(n_anterior, n_neuronas))
            bloques.append(nn.ReLU())
            bloques.append(nn.Dropout(dropout))
            n_anterior = n_neuronas
        bloques.append(nn.Linear(n_anterior, 1))
        bloques.append(nn.Sigmoid())
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x)


def load_public_links():
    """
    Carga los enlaces públicos del proyecto desde `docs/materiales_publicos.json`.
    Si el archivo no existe o está incompleto, la app sigue funcionando y
    muestra una guía para completar los enlaces finales.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    links_path = os.path.join(base_dir, "docs", "materiales_publicos.json")

    defaults = {
        "reporte": {
            "label": "Reporte técnico",
            "url": "",
            "description": "Entrada de blog o publicación final del reporte técnico.",
        },
        "video": {
            "label": "Video promocional",
            "url": "",
            "description": "Video publicitario de la aplicación web.",
        },
    }

    if not os.path.exists(links_path):
        return defaults

    try:
        with open(links_path, "r", encoding="utf-8") as f:
            custom_links = json.load(f)
    except (OSError, json.JSONDecodeError):
        return defaults

    for key in defaults:
        if isinstance(custom_links.get(key), dict):
            defaults[key].update(custom_links[key])

    return defaults

# ─── Carga de Artefactos ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    base_dir      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path    = os.path.join(base_dir, "models",  "modelo_final.pt")
    scaler_path   = os.path.join(base_dir, "models",  "scaler.pkl")
    woe_path      = os.path.join(base_dir, "outputs", "woe_bins.pkl")
    params_path   = os.path.join(base_dir, "outputs", "scorecard_params.pkl")
    defaults_path = os.path.join(base_dir, "outputs", "defaults.json")
    scores_path   = os.path.join(base_dir, "outputs", "distribucion_scores.csv")

    # Archivos obligatorios
    required = [model_path, scaler_path, woe_path, params_path, defaults_path]
    if not all(os.path.exists(p) for p in required):
        return None

    # El scaler solo se utiliza para recuperar el orden canónico de las features
    # (fue fiteado para la regresión logística — ver Notebook 03)
    scaler           = joblib.load(scaler_path)
    feature_names    = list(scaler.feature_names_in_)

    woe_bins         = joblib.load(woe_path)
    scorecard_params = joblib.load(params_path)

    # Carga de los valores por defecto (medianas/modas reales del dataset)
    with open(defaults_path, "r", encoding="utf-8") as f:
        defaults = json.load(f)

    # Carga de la distribución real de scores (opcional pero recomendada)
    if os.path.exists(scores_path):
        df_scores_pop = pd.read_csv(scores_path)
    else:
        df_scores_pop = None

    # Carga del modelo — arquitectura fija (128→64→32, dropout 0.4)
    n_entrada  = len(feature_names)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)

    model = RedNeuronal(n_entrada, capas=[128, 64, 32], dropout=0.4)
    model.load_state_dict(state_dict)
    model.eval()

    return model, feature_names, woe_bins, scorecard_params, defaults, df_scores_pop


artifacts = load_artifacts()
PUBLIC_LINKS = load_public_links()

if artifacts is None:
    st.error("### ⚠️ Archivos necesarios no encontrados")
    st.markdown("""
    Para que la aplicación funcione, ejecuta primero los Notebooks **02**, **03** y **04**
    y asegúrate de que estos archivos existan:

    - `models/modelo_final.pt`
    - `models/scaler.pkl`
    - `outputs/woe_bins.pkl`
    - `outputs/scorecard_params.pkl`
    - `outputs/defaults.json` *(generado por el script `generar_defaults.py`)*
    - `outputs/distribucion_scores.csv` *(opcional — mejora la visualización de población)*
    """)
    st.stop()

model, FEATURE_NAMES, woe_bins, scorecard_params, DEFAULTS, df_scores_pop = artifacts

# ─── Parámetros de la Scorecard ───────────────────────────────────────────────
SCORE_BASE = scorecard_params.get("score_base",    600)
ODDS_BASE  = scorecard_params.get("odds_objetivo", 1)
PDO        = scorecard_params.get("pdo",           50)
FACTOR     = PDO / np.log(2)
OFFSET     = SCORE_BASE - FACTOR * np.log(max(ODDS_BASE, 1e-9))
SCORE_MIN, SCORE_MAX = 300, 850

BANDAS = [
    ("A", "Riesgo muy bajo", 700, 850, "#10B981"),
    ("B", "Riesgo bajo",     650, 699, "#34D399"),
    ("C", "Riesgo medio",    600, 649, "#FBBF24"),
    ("D", "Riesgo alto",     550, 599, "#F97316"),
    ("E", "Riesgo muy alto", 300, 549, "#EF4444"),
]

# ─── Funciones ────────────────────────────────────────────────────────────────

def banda_de(score: int):
    for letra, desc, lo, hi, color in BANDAS:
        if lo <= score <= hi:
            return letra, desc, color
    return "E", "Riesgo muy alto", "#EF4444"


def calcular_score(datos: dict):
    """
    Pipeline: variables brutas → WOE → Red Neuronal → Score PDO.

    Importante: la Red Neuronal fue entrenada directamente sobre los valores
    WOE brutos (ver Notebook 03). No se aplica StandardScaler aquí.
    """
    # Completar con valores por defecto (medianas/modas reales del dataset)
    datos_completos = {**DEFAULTS, **datos}
    df_input = pd.DataFrame([datos_completos])

    # 1. Transformación WOE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_woe = sc.woebin_ply(df_input, woe_bins)

    # Reordenar las columnas en el orden esperado por el modelo
    for col in FEATURE_NAMES:
        if col not in df_woe.columns:
            df_woe[col] = 0.0

    df_modelo = df_woe[FEATURE_NAMES]

    # Factores de impacto para el análisis de variables
    factores = []
    for col in FEATURE_NAMES:
        woe_val    = df_modelo[col].iloc[0]
        pts_impact = int(round(-woe_val * FACTOR))
        if pts_impact != 0:
            nom_base = col.replace("_woe", "")
            val = datos_completos.get(nom_base, "N/A")
            factores.append({"nombre": nom_base, "valor": val, "impacto": pts_impact})

    # 2. Red Neuronal directamente sobre los valores WOE (sin scaler)
    x_tensor = torch.FloatTensor(df_modelo.values)
    with torch.no_grad():
        prob = model(x_tensor).item()

    # 3. Recalibración — corrección del sesgo por undersampling 50/50
    prior_train = 0.50
    prior_real  = 0.219
    odds_raw    = prob / (1 - prob)
    odds_cal    = odds_raw * (prior_real / (1 - prior_real)) / (prior_train / (1 - prior_train))
    prob        = odds_cal / (1 + odds_cal)

    # 4. Conversión a score PDO
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    odds = (1 - prob) / prob
    s    = OFFSET + FACTOR * np.log(odds)

    return int(max(SCORE_MIN, min(SCORE_MAX, round(s)))), prob, factores


# ─── Gráficas ─────────────────────────────────────────────────────────────────

def gauge_chart(score: int) -> go.Figure:
    _, _, color = banda_de(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 64, "color": color, "family": "Inter, sans-serif"},
                "suffix": " pts"},
        gauge={
            "axis": {"range": [SCORE_MIN, SCORE_MAX], "tickwidth": 1,
                     "tickcolor": "#CBD5E1", "dtick": 50,
                     "tickfont": {"size": 10, "color": "#94A3B8"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#F8FAFC",
            "borderwidth": 0,
            "steps": [
                {"range": [300, 549], "color": "#FEE2E2"},
                {"range": [550, 599], "color": "#FED7AA"},
                {"range": [600, 649], "color": "#FEF3C7"},
                {"range": [650, 699], "color": "#D1FAE5"},
                {"range": [700, 850], "color": "#A7F3D0"},
            ],
            "threshold": {"line": {"color": color, "width": 5},
                          "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=25, r=25, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def population_chart(score_usuario: int):
    """
    Histograma de la distribución real de scores del conjunto de test.
    Si distribucion_scores.csv no está disponible, se recurre a una
    distribución simulada (fallback de último recurso).
    """
    if df_scores_pop is not None and "score" in df_scores_pop.columns:
        pop = df_scores_pop["score"].to_numpy()
        fuente_label = "Población real (conjunto de test)"
    else:
        # Fallback simulado — solo si el CSV no existe
        np.random.seed(42)
        pop = np.concatenate([
            np.random.normal(660, 55, 7500),
            np.random.normal(490, 45, 2500),
        ])
        pop = np.clip(pop, SCORE_MIN, SCORE_MAX)
        fuente_label = "Distribución simulada"

    percentil = (pop <= score_usuario).sum() / len(pop) * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pop, nbinsx=60,
        marker_color="#CBD5E1", marker_line_color="#E2E8F0",
        marker_line_width=0.5, opacity=0.85,
        histnorm="probability density", name=fuente_label,
        hovertemplate="Score: %{x}<br>Densidad: %{y:.4f}<extra></extra>",
    ))
    _, _, color = banda_de(score_usuario)
    fig.add_vline(
        x=score_usuario, line_width=3, line_color=color,
        annotation_text=f"  Tú: {score_usuario} pts  (P{percentil:.0f})",
        annotation_position="top right",
        annotation_font=dict(size=13, color=color, family="Inter, sans-serif"),
        annotation_bgcolor="white", annotation_bordercolor=color,
        annotation_borderwidth=1, annotation_borderpad=4,
    )
    fig.update_layout(
        height=280,
        xaxis_title=f"Score crediticio — {fuente_label}", yaxis_title="",
        yaxis_showticklabels=False, showlegend=False,
        margin=dict(l=15, r=15, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#F1F5F9", showgrid=True),
        yaxis=dict(gridcolor="#F1F5F9", showgrid=False),
        font={"family": "Inter, sans-serif"},
    )
    return fig, percentil


def render_public_materials(links: dict, *, compact: bool = False):
    """
    Renderiza accesos al reporte técnico y al video promocional.
    """
    title = "### Material del proyecto" if compact else "### Explora el proyecto completo"
    st.markdown(title)

    missing = []
    for key, title_text in [("reporte", "Reporte técnico"), ("video", "Video promocional")]:
        item = links.get(key, {})
        label = item.get("label", title_text)
        url = (item.get("url") or "").strip()
        if key == "video":
            url = https://youtube.com/shorts/lDyEkjEELtQ
        description = item.get("description", "")

        st.markdown(f"**{label}**")
        if description:
            st.caption(description)
    
        if url:
            st.link_button(f"Abrir {label}", url, use_container_width=True)
        else:
            missing.append(label)
            st.info(f"Pendiente por configurar: agrega la URL de {label.lower()} en `docs/materiales_publicos.json`.")

    if missing and not compact:
        st.caption(
            "La aplicación ya incluye el espacio para estos entregables. "
            "Solo falta reemplazar las URLs vacías por las finales."
        )


# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    footer, #MainMenu { visibility: hidden; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] label { font-weight: 500; }
    section[data-testid="stSidebar"] .info-box,
    section[data-testid="stSidebar"] .info-box * { color: #1E1B4B !important; }
    .hero { text-align: center; padding: 1.2rem 0 0.5rem; }
    .hero h1 {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #6366F1, #8B5CF6, #A78BFA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p { color: #64748B; font-size: 1rem; margin: 0; }
    .kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
    .kpi-card { border-radius: 16px; padding: 1.4rem 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
    .kpi-card .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: .06em; opacity: .85; margin: 0; }
    .kpi-card .value { font-size: 2.4rem; font-weight: 700; margin: 0.3rem 0 0; line-height: 1; }
    .kpi-card .sub   { font-size: 0.78rem; opacity: .7; margin: 0; }
    .section-title { font-size: 1.05rem; font-weight: 600; color: #334155; border-bottom: 2px solid #E2E8F0; padding-bottom: 0.4rem; margin: 1.5rem 0 0.8rem; }
    .bandas-strip { display: flex; gap: 4px; margin: 0.5rem 0; }
    .banda-cell { flex: 1; text-align: center; padding: 0.6rem 0.3rem; border-radius: 10px; color: white; font-weight: 600; font-size: 0.85rem; }
    .banda-cell .letra { font-size: 1.4rem; display: block; }
    .banda-cell .rango { font-size: 0.7rem; opacity: .85; }
    .factor-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.55rem 0.8rem; border-radius: 8px; margin-bottom: 0.4rem; font-size: 0.9rem; }
    .factor-pos { background: #F0FDF4; border-left: 3px solid #10B981; }
    .factor-neg { background: #FEF2F2; border-left: 3px solid #EF4444; }
    .factor-icon { font-size: 1.1rem; flex-shrink: 0; }
    .factor-text { flex: 1; color: #334155; }
    .factor-pts  { font-weight: 700; min-width: 50px; text-align: right; }
    .factor-pts.pos { color: #10B981; }
    .factor-pts.neg { color: #EF4444; }
    .disclaimer { text-align: center; color: #94A3B8; font-size: 0.75rem; padding: 1rem 0 0.5rem; border-top: 1px solid #F1F5F9; margin-top: 2rem; }
    .info-box { background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.8rem; color: #1E1B4B !important; margin-top: 1rem; line-height: 1.6; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar — Formulario ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Tu Solicitud")

    st.markdown("#### Préstamo")
    loan_amnt = st.number_input(
        "Monto solicitado (USD)",
        min_value=500, max_value=35_000, value=13_000, step=500,
        help="loan_amnt · Rango del dataset: 500 – 35 000 USD · Mediana: 13 000 USD",
    )
    term = st.selectbox(
        "Plazo de devolución",
        [" 36 months", " 60 months"],
        format_func=lambda x: "36 meses" if "36" in x else "60 meses",
        help="term · Variable más importante según SHAP (importancia: 0.076)",
    )

    st.markdown("#### Perfil financiero")
    annual_inc = st.number_input(
        "Ingreso anual declarado (USD)",
        min_value=0, max_value=500_000, value=60_000, step=5_000,
        help="annual_inc · 2ª variable más importante según SHAP (importancia: 0.049)",
    )
    tot_cur_bal = st.number_input(
        "Saldo total en todas tus cuentas (USD)",
        min_value=0, max_value=500_000, value=40_000, step=1_000,
        help="tot_cur_bal · Suma de los saldos actuales en todas tus líneas de crédito · 3ª variable más importante según SHAP (importancia: 0.041) · Mediana del dataset: 80 559 USD",
    )

    st.markdown("""
    <div class="info-box" style="color: #1E1B4B !important;">
    ℹ️ Las demás variables del modelo se fijan automáticamente a los valores medianos del dataset,
    ya que no son directamente conocidas por el solicitante en el momento de la solicitud.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    render_public_materials(PUBLIC_LINKS, compact=True)

    st.markdown("---")
    calcular = st.button("Calcular mi Score", width="stretch", type="primary")


# ─── Contenido principal ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>CreditScore</h1>
    <p>Estima tu puntaje crediticio con solo 4 datos básicos</p>
</div>
""", unsafe_allow_html=True)

if not calcular:
    st.markdown("""
    <div class="bandas-strip">
        <div class="banda-cell" style="background:#10B981"><span class="letra">A</span><span class="rango">700 – 850</span></div>
        <div class="banda-cell" style="background:#34D399"><span class="letra">B</span><span class="rango">650 – 699</span></div>
        <div class="banda-cell" style="background:#FBBF24"><span class="letra">C</span><span class="rango">600 – 649</span></div>
        <div class="banda-cell" style="background:#F97316"><span class="letra">D</span><span class="rango">550 – 599</span></div>
        <div class="banda-cell" style="background:#EF4444"><span class="letra">E</span><span class="rango">300 – 549</span></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### ¿Cómo funciona?
        1. **Completa 4 datos básicos** en el panel lateral izquierdo.
        2. **Haz clic en "Calcular mi Score"** para obtener tu resultado.
        3. **Revisa tu perfil** : score, probabilidad de incumplimiento,
           banda de riesgo y tu posición frente a la población.

        ### Sobre el modelo
        Este score está calculado en **tiempo real** por una **Red Neuronal Artificial**
        entrenada sobre 268 530 créditos históricos de LendingClub.
        El formulario solicita únicamente las **3 variables más importantes según SHAP**
        (`term`, `annual_inc`, `tot_cur_bal`) más el monto solicitado (`loan_amnt`).
        Las variables restantes del modelo se fijan a los valores medianos del dataset.
        """)
    with c2:
        st.markdown("""
        ### Escala
        | Banda | Nivel |
        |:---:|---|
        | **A** | Riesgo muy bajo |
        | **B** | Riesgo bajo |
        | **C** | Riesgo medio |
        | **D** | Riesgo alto |
        | **E** | Riesgo muy alto |

        > *Modelo PyTorch · WOE · Escala PDO*
        """)

        st.markdown("""
        ### Variables del modelo
        | Rang SHAP | Variable |
        |:---:|---|
        | 🥇 1 | `term` |
        | 🥈 2 | `annual_inc` |
        | 🥉 3 | `tot_cur_bal` |
        | 4 | `dti` * |
        | 5 | `revol_util` * |
        | 6 | `installment` * |
        | 7 | `total_rev_hi_lim` * |
        | 8 | `loan_amnt` |
        | 9 | `verification_status` * |

        *\* fijado a la mediana*
        """)

    st.markdown("---")
    render_public_materials(PUBLIC_LINKS)

else:
    datos_dict = {
        "loan_amnt" : loan_amnt,
        "term"      : term,
        "annual_inc": annual_inc,
        "tot_cur_bal": tot_cur_bal,
    }

    score, prob, factores = calcular_score(datos_dict)
    letra, desc_banda, color_banda = banda_de(score)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card" style="background: linear-gradient(135deg, #EEF2FF, #E0E7FF);">
            <p class="label" style="color:#6366F1;">Score crediticio</p>
            <p class="value" style="color:#4338CA;">{score}</p>
            <p class="sub"   style="color:#6366F1;">de {SCORE_MAX} puntos</p>
        </div>
        <div class="kpi-card" style="background: linear-gradient(135deg, #FFF1F2, #FFE4E6);">
            <p class="label" style="color:#E11D48;">Probabilidad de default</p>
            <p class="value" style="color:#BE123C;">{prob*100:.1f}%</p>
            <p class="sub"   style="color:#E11D48;">estimada por la Red Neuronal</p>
        </div>
        <div class="kpi-card" style="background:{color_banda}22;">
            <p class="label" style="color:{color_banda};">Banda de riesgo</p>
            <p class="value" style="color:{color_banda};">{letra}</p>
            <p class="sub"   style="color:{color_banda};">{desc_banda}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_g, col_p = st.columns(2)
    with col_g:
        st.markdown('<p class="section-title">Indicador de Score</p>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(score), width="stretch", config={"displayModeBar": False})
    with col_p:
        st.markdown('<p class="section-title">Posición en la población</p>', unsafe_allow_html=True)
        fig_pop, perc = population_chart(score)
        st.plotly_chart(fig_pop, width="stretch", config={"displayModeBar": False})

    st.markdown('<p class="section-title">Impacto de cada variable sobre tu score</p>',
                unsafe_allow_html=True)

    # Variables visibles saisies par le client (les seules à afficher)
    VARS_CLIENT = {"loan_amnt", "term", "annual_inc", "tot_cur_bal"}
    NOMS_VARIABLES = {
        "loan_amnt"  : "Monto solicitado",
        "term"       : "Plazo del préstamo",
        "annual_inc" : "Ingreso anual",
        "tot_cur_bal": "Saldo total en cuentas activas",
    }

    factores_sorted = sorted(factores, key=lambda x: x["impacto"])
    rows_html = ""
    for f in factores_sorted:
        if f["nombre"] not in VARS_CLIENT:
            continue
        cls   = "factor-pos" if f["impacto"] > 0 else "factor-neg"
        icon  = "✅" if f["impacto"] > 0 else "⚠️"
        pcls  = "pos" if f["impacto"] > 0 else "neg"
        label = f'{NOMS_VARIABLES.get(f["nombre"], f["nombre"])} ({f["nombre"]}) — {f["valor"]}'
        rows_html += (
            f'<div class="factor-row {cls}">'
            f'<span class="factor-icon">{icon}</span>'
            f'<span class="factor-text">{label}</span>'
            f'<span class="factor-pts {pcls}">{f["impacto"]:+d} pts</span>'
            f'</div>'
        )
    st.markdown(rows_html, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    Modelo de Redes Neuronales · PyTorch · Score PDO (Siddiqi, 2006) · WOE
    · <a href="https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data"
       target="_blank" style="color:#6366F1;">Credit Risk Dataset — Kaggle</a>
</div>
""", unsafe_allow_html=True)
