import streamlit as st
import pandas as pd

from utils.integrantes import dict_integrantes

def info_integrantes():
    st.subheader("Integrantes CodeX")

    dados = []

    for integrante, rm in dict_integrantes.items():
        dados.append({
            "Integrante": integrante,
            "RM": rm
        })

    df = pd.DataFrame(dados)
    st.table(df)

def info_pipeline():
    st.markdown("""
    ### Pipeline do Sistema

    1️⃣ Upload da imagem  
    2️⃣ Pré-processamento  
    3️⃣ Codificação (Encoder)  
    4️⃣ Espaço latente  
    5️⃣ Reconstrução (Decoder)  
    6️⃣ Cálculo do erro  
    7️⃣ Classificação
    """)

def gerar_tab_sobre(vae):

    info_integrantes()
    st.divider()
    info_pipeline()
    st.divider()

    st.header("📊 Sobre o Modelo VAE")
    st.markdown("""
        ### Arquitetura do Modelo

        **Encoder:**
        - Conv2D(32) → Conv2D(64) → Flatten → Dense(128) → Latent Space

        **Decoder:**
        - Dense(7×7×64) → Reshape → Conv2DTranspose(64) → Conv2DTranspose(32) → Output

        **Dimensão Latente:** 16 variáveis

        ### Como Funciona a Triagem

        1. **Imagens Normais:** Baixo erro de reconstrução (padrão bem aprendido)
        2. **Imagens com Pneumonia:** Alto erro de reconstrução (padrão diferente)
        3. **Thresholds:** - < 0.01: Normal
           - 0.01-0.02: Borderline  
           - > 0.02: Possível pneumonia

        ### Limitações

        - Treinado apenas em PneumoniaMNIST
        - Não substitui diagnóstico médico
        - Sensibilidade depende da qualidade da imagem
        """)

    if vae:
        st.subheader("Estatísticas do Modelo")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parâmetros Encoder", f"{vae.encoder.count_params():,}")
            st.metric("Parâmetros Decoder", f"{vae.decoder.count_params():,}")
        with col2:
            st.metric("Total Parâmetros", f"{vae.count_params():,}")
            st.metric("Dimensão Latente", vae.encoder.output_shape[0][-1])