import streamlit as st
from modelo import generate_new_images
from PIL import Image

def gerar_tab_imagens(vae):
    st.header("Geração de Novas Imagens de Raio-X")
    st.markdown("""
                Gere novas imagens sintéticas de raio-X usando o espaço latente aprendido pelo VAE.
                """)

    col1, col2 = st.columns([2, 1])

    with col1:
        num_images = st.slider("Número de imagens a gerar", 1, 8, 4)

        if st.button("Gerar Novas Imagens", type="primary"):
            with st.status("Gerando imagens...", expanded=True) as status:
                st.write("Amostrando vetores...")

                generated = generate_new_images(vae, num_images)
                st.session_state.generated_images = generated

                st.write("✅ Reconstruindo imagens pelo Decoder...")

                status.update(label="Imagens geradas com sucesso!", state="complete", expanded=False)

    with col2:
        st.markdown("""
                    **Controles:**
                    - Ajuste o número de imagens
                    - Clique em gerar para criar novas
                    - As imagens são amostradas do espaço latente normal
                    """)

    if 'generated_images' in st.session_state:
        st.subheader("Imagens Geradas")

        qtd_gerada = len(st.session_state.generated_images)

        cols = st.columns(qtd_gerada)
        for i, col in enumerate(cols):
            with col:
                st.image(st.session_state.generated_images[i].squeeze(),
                         clamp=True,
                         caption=f"Imagem {i + 1}",
                         use_column_width=True)

        if st.button("Salvar Imagens"):
            images = []
            for i in range(qtd_gerada):
                img_array = (st.session_state.generated_images[i].squeeze() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                images.append(img)

            st.success(f"Imagens salvas na lista!.")