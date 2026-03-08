import streamlit as st
import pandas as pd

def gerar_tab_historico():
    st.header("📋 Histórico de Avaliações e Feedback")

    st.markdown("Acompanhe o desempenho do modelo com base no feedback.")

    if len(st.session_state.historico) > 0:
        df_historico = pd.DataFrame(st.session_state.historico)

        total_avaliacoes = len(df_historico)
        acertos = len(df_historico[df_historico["Feedback"] == "Acerto"])
        taxa_acerto = (acertos / total_avaliacoes) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("Total de Avaliações", total_avaliacoes)
        col2.metric("Feedbacks Positivos", acertos)
        col3.metric("Taxa de Acerto do Modelo", f"{taxa_acerto:.1f}%")

        if total_avaliacoes >= 3 and taxa_acerto <= 60.0:
            st.error("⚠️ **ALERTA DE DEGRADAÇÃO:** O modelo apresenta taxa de acerto igual ou inferior a 60% nas "
            "últimas avaliações.")
        elif total_avaliacoes >= 3 and taxa_acerto < 80.0:
            st.warning("⚠️ **Atenção:** A taxa de acerto está caindo. Monitore as próximas avaliações.")

        st.divider()

        st.dataframe(
            df_historico,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Erro": st.column_config.NumberColumn(
                    "MSE (Erro)",
                    format="%.5f"
                ),
                "Confiança": st.column_config.ProgressColumn(
                    "Nível de Confiança",
                    help="Grau de certeza da predição do modelo",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Feedback": st.column_config.TextColumn(
                    "Avaliação do Especialista"
                )
            }
        )

        if st.button("🗑️ Limpar Histórico"):
            st.session_state.historico = []
            st.rerun()

    else:
        st.info(
            "Nenhuma avaliação registrada ainda. Vá na aba 'Triagem' e forneça feedback sobre as análises para preencher "
        "esta tabela.")
