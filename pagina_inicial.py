import streamlit as st
from modelo import load_model
from ui.sidebar import sidebar
from ui.tab_sobre import gerar_tab_sobre
from ui.tab_historico import gerar_tab_historico
from ui.tab_imagens import gerar_tab_imagens
from ui.tab_home import gerar_tab_home

st.set_page_config(page_title="Checkpoint", layout="wide", initial_sidebar_state="expanded", page_icon="✖")

@st.cache_resource
def get_model():
    vae, err = load_model()
    return vae, err

def limpar_analise():
    st.session_state.recon = None
    st.session_state.analise_concluida = False

def pagina_inicial():
    if "historico" not in st.session_state:
        st.session_state.historico = []
    if "analise_concluida" not in st.session_state:
        st.session_state.analise_concluida = False
    if "recon" not in st.session_state:
        st.session_state.recon = None

    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 22px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Generative AI Advanced Net & Front End e Mobile Development")

    vae, err = get_model()

    sidebar(vae, err)

    if err:
        st.error(err)
        st.stop()

    tab_home, tab_imagens, tab_historico, tab_sobre = st.tabs(["Triagem", "Gerar Imagens", "Histórico", "Sobre"])

    with tab_home:
        gerar_tab_home(vae)

    with tab_imagens:
        gerar_tab_imagens(vae)

    with tab_historico:
        gerar_tab_historico()

    with tab_sobre:
        gerar_tab_sobre(vae)

if __name__ == "__main__":
    pagina_inicial()
