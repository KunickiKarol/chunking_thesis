import os
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.evaluate_generator.evaluate_generator_one import evaluate_generator_one
from src.tools.presets import iter_cfg_with_presets


def generation_all(
    datasets_cfg,
    chunking_cfg,
    splits,
    embed_cfg,
    search_cfg,
    rerank_cfg,
    generation_cfg,
    evaluator_cfg,
    dataset_dir: Path,
    generation_dir: Path,
    evaluator_dir: Path,
):
    print(f"➡️ Generation: {generation_dir}")
    for (
        (dataset_name, dataset_preset),
        (chunking_name, chunking_preset),
        (embed_name, embed_preset),
        (search_name, search_preset),
        (rerank_name, rerank_preset),
        (generation_name, generation_preset),
        (evaluator_name, evaluator_preset),
        split,
    ) in product(
        iter_cfg_with_presets(datasets_cfg),
        iter_cfg_with_presets(chunking_cfg),
        iter_cfg_with_presets(embed_cfg),
        iter_cfg_with_presets(search_cfg),
        iter_cfg_with_presets(rerank_cfg),
        iter_cfg_with_presets(generation_cfg),
        iter_cfg_with_presets(evaluator_cfg),
        splits,
    ):

        dataset_preset_name = dataset_preset["name"]
        chunking_preset_name = chunking_preset["name"]
        embed_preset_name = embed_preset["name"]
        search_preset_name = search_preset["name"]
        generation_preset_name = generation_preset["name"]
        rerank_preset_name = rerank_preset["name"]
        evaluator_preset_name = evaluator_preset["name"]

        task_type = dataset_preset["params"]["task_type"]
        generation_preset_params = generation_preset["params"]

        task_input_dir = dataset_dir / dataset_name / dataset_preset_name / "Tasks" / task_type / split

        generation_input_dir = (
            generation_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / embed_name
            / embed_preset_name
            / search_name
            / search_preset_name
            / rerank_name
            / rerank_preset_name
            / generation_name
            / generation_preset_name
            / split
        )

        result_dir = (
            evaluator_dir
            / dataset_name
            / dataset_preset_name
            / chunking_name
            / chunking_preset_name
            / embed_name
            / embed_preset_name
            / search_name
            / search_preset_name
            / rerank_name
            / rerank_preset_name
            / generation_name
            / generation_preset_name
            / evaluator_name
            / evaluator_preset_name
            / split
        )

        if not task_input_dir.exists():
            print(f"❌ Brak zadań: {task_input_dir}, pomijam...")
            continue

        if not generation_input_dir.exists():
            print(f"❌ Brak generacji: {generation_input_dir}, pomijam...")
            continue

        if result_dir.exists() and any(p.is_file() for p in result_dir.iterdir()):
            print(f"⏭️ Pomijam {result_dir}, generation już istnieje")
            continue

        result_dir.mkdir(parents=True, exist_ok=True)

        evaluate_generator_one(
            generation_name,
            generation_preset_params,
            task_input_dir,
            generation_input_dir,
            result_dir,
        )


def main():
    load_dotenv()

    DATASET_DIR = Path(os.getenv("DATASETS_DIR"))
    GENERATION_DIR = Path(os.getenv("GENERATION_DIR"))
    EVALUATION_DIR = Path(os.getenv("EVALUATION_DIR"))
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    datasets_cfg = params.get("preprocess_datasets").get("datasets")
    if not datasets_cfg:
        raise ValueError("Nie znaleziono datasets_download w params.yaml")

    chunking_cfg = params.get("chunking").get("methods")
    if not chunking_cfg:
        raise ValueError("Nie znaleziono chunking_methods w params.yaml")

    embed_cfg = params.get("vector_embed").get("methods")
    if not embed_cfg:
        raise ValueError("Nie znaleziono vector_embed_methods w params.yaml")

    search_cfg = params.get("search").get("methods")
    if not search_cfg:
        raise ValueError("Nie znaleziono search_methods w params.yaml")

    rerank_cfg = params.get("rerank").get("methods")
    if not rerank_cfg:
        raise ValueError("Nie znaleziono rerank_methods w params.yaml")

    generation_cfg = params.get("generation").get("methods")
    if not generation_cfg:
        raise ValueError("Nie znaleziono generation_methods w params.yaml")
    
    evaluator_cfg = params.get("evaluation").get("methods")
    if not evaluator_cfg:
        raise ValueError("Nie znaleziono evaluation_methods w params.yaml")

    splits = params.get("chunking").get("splits")
    if not splits:
        raise ValueError("Nie znaleziono splits w params.yaml")

    generation_all(
        datasets_cfg=datasets_cfg,
        chunking_cfg=chunking_cfg,
        splits=splits,
        embed_cfg=embed_cfg,
        search_cfg=search_cfg,
        rerank_cfg=rerank_cfg,
        generation_cfg=generation_cfg,
        evaluator_cfg=evaluator_cfg,
        dataset_dir=DATASET_DIR,
        generation_dir=GENERATION_DIR,
        evaluator_dir=EVALUATION_DIR,
    )


if __name__ == "__main__":
    main()
