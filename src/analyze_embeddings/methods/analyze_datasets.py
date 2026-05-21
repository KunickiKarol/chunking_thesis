from src.analyze_embeddings.methods.register import register_analyze_embeddings


@register_analyze_embeddings("analyze_datasets")
def analyze_datasets(analyze_preset_params, result_dir, df_embedding, df_bookmeta):
    df_merged = df_embedding.join(
    df_bookmeta,
    how="left",
    validate="many_to_one"
)