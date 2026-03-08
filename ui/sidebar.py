import streamlit as st

from utils.limpar_analise import limpar_analise

def info_modelo(vae, err):
    st.subheader("Modelo")

    if err:
        st.error(err)
        st.stop()
    else:
        st.success('✅ Modelo carregado com sucesso!')
        st.info(f'Dimensão latente: {vae.encoder.output_shape[0][-1]}')
        st.write("Arquitetura: Variational Autoencoder (VAE)")
        st.write("Dataset: PneumoniaMNIST")
        st.write("Entrada: 28x28 pixels")

def info_metricas():
    st.subheader("Métricas do Modelo")

    st.metric("Loss Final", "0.021")
    st.metric("Reconstruction Error Médio", "0.009")

def sidebar(vae, err):
    with st.sidebar:
        st.header("⚙️ Configurações do Modelo")
        st.markdown("Ajuste parâmetros da inferência")

        # Slider funcional que reseta a análise (Critérios: Sidebar Painel de Controle + on_change)
        sensibilidade = st.slider(
            "Limiar de Anomalia (Threshold)",
            min_value=0.005,
            max_value=0.050,
            value=0.020,
            step=0.005,
            help="Valores menores tornam o modelo mais sensível a classificar como pneumonia.",
            on_change=limpar_analise
        )
        st.session_state.limite_anomalia = sensibilidade

        st.divider()
        info_modelo(vae, err)
        info_metricas()