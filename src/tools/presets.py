def load_presets(presets_file, use_presets):
    """Ładuje YAML presetów i filtruje wg use_presets"""
    with open(presets_file) as f:
        all_presets = yaml.safe_load(f)["presets"]

    if use_presets:
        presets = [p for p in all_presets if p["name"] in use_presets]
        missing = set(use_presets) - {p["name"] for p in presets}
        if missing:
            raise ValueError(f"Nie znaleziono presetów: {missing}")
        return presets
    return all_presets
