import streamlit as st
import io
from PIL import Image

from utils.limpar_analise import limpar_analise
from modelo import preprocess_image, compute_reconstruction_error, classify_pneumonia

def gerar_tab_home(vae):
    st.header("Triagem de Pneumonia")

    uploaded = st.file_uploader(
        'Envie uma imagem de raio-X para análise',
        type=['png', 'jpg', 'jpeg'],
        key='upload_triagem',
        on_change=limpar_analise
    )

    if uploaded:
        enviar = st.button("Enviar", type="primary")

        if enviar:
            with st.status("Processando raio-X...", expanded=True) as status:
                barra_progresso = st.progress(0, text="Iniciando extração de características...")

                image = Image.open(io.BytesIO(uploaded.read()))
                x = preprocess_image(image)

                barra_progresso.progress(40, text="Reconstruindo pelo VAE...")
                st.session_state.recon = vae(x, training=False).numpy()

                barra_progresso.progress(80, text="Calculando erro e classificando...")
                mse = compute_reconstruction_error(x, st.session_state["recon"])
                classification, description, color = classify_pneumonia(mse, st.session_state.limite_anomalia)

                barra_progresso.progress(100, text="Finalizando...")

                st.session_state.img_original = image
                st.session_state.mse = mse
                st.session_state.classification = classification
                st.session_state.confianca = (1 - mse) * 100 if mse < 1 else 0
                st.session_state.analise_concluida = True

                status.update(label="Análise Concluída com Sucesso!", state="complete", expanded=False)

        if st.session_state.get("analise_concluida"):
            col_input, col_output = st.columns(2)

            with col_input:
                st.subheader("Imagem Original")
                st.image(st.session_state.img_original, width=400, use_column_width=True)

            with col_output:
                st.subheader("Reconstrução VAE")
                st.image(
                    st.session_state.recon[0].squeeze(),
                    width=400,
                    clamp=True,
                    use_column_width=True
                )

            st.sidebar.divider()
            st.sidebar.subheader("Resultado da Análise")
            st.sidebar.metric("Erro de Reconstrução", f"{st.session_state.mse:.6f}")
            st.sidebar.metric("Classificação", st.session_state.classification)

            conf = st.session_state.confianca
            st.sidebar.metric("Confiança", f"{conf:.1f}%")

            if conf >= 90:
                st.sidebar.success("Alta confiança na predição.")
            elif conf >= 70:
                st.sidebar.warning("Confiança moderada. Recomendada revisão atenta do raio-X.")
            else:
                st.sidebar.error("Baixa confiança. O resultado pode ser inconclusivo.")

            st.sidebar.caption("⚠️ Nota: Este modelo fornece uma estimativa e não substitui diagnóstico médico.")

            st.divider()
            st.subheader("Feedback do Especialista")
            st.write("A classificação do modelo condiz com a avaliação clínica?")

            col_fb1, col_fb2 = st.columns([1, 1])

            if col_fb1.button("👍 Sim, modelo acertou", use_container_width=True):
                st.session_state.historico.append({
                    "Erro": st.session_state.mse,
                    "Classificação": st.session_state.classification,
                    "Confiança": st.session_state.confianca,
                    "Feedback": "Acerto"
                })
                st.toast("✅ Feedback positivo registrado!")

            if col_fb2.button("👎 Não, modelo errou", use_container_width=True):
                st.session_state.historico.append({
                    "Erro": st.session_state.mse,
                    "Classificação": st.session_state.classification,
                    "Confiança": st.session_state.confianca,
                    "Feedback": "Erro"
                })
                st.toast("Feedback negativo registrado para calibração futura!")

    else:
        st.info("👆 Envie uma imagem de raio-X para iniciar a análise.")