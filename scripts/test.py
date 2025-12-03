import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import yaml
from pytorch_lightning import Trainer

from scripts.utils import setup_reproducibility, shut_down_warnings
from src.data import get_test_dataloader 
from src.models.utils import build_model_from_config
from src.training import MelanomaLitModule

setup_reproducibility(seed=27)
shut_down_warnings()

def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Inference / Prediction CLI")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file used in training.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the .ckpt file (best or last).")
    parser.add_argument(
        "--output-csv", type=str, default="submission.csv", help="Name of the output CSV file.")

    args = parser.parse_args()
    
    # ----------------------------------------------------
    # 1. Load Config & Setup
    # ----------------------------------------------------
    config_path = Path(args.config)
    config = load_config(config_path)

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    aug_cfg_path = data_cfg.get("aug_cfg_path", None)
    if aug_cfg_path is not None:
        aug_cfg_path = Path(aug_cfg_path)
        aug_cfg = load_config(aug_cfg_path)
    else:
        aug_cfg = {}

    # ----------------------------------------------------
    # 2. Data
    # ----------------------------------------------------
    test_loader = get_test_dataloader(
        aug_cfg=aug_cfg,
        batch_size=data_cfg.get("batch_size", 256),
        num_workers=data_cfg.get("num_workers", 4),
    )

    # ----------------------------------------------------
    # 3. Model
    # ----------------------------------------------------
    backbone = build_model_from_config(model_cfg)
    
    # Carregamos os pesos do checkpoint para dentro do LightningModule
    print(f"Carregando checkpoint de: {args.checkpoint}")

    lit_model = MelanomaLitModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=backbone,
        training_cfg=training_cfg, # Passamos a config para manter compatibilidade
        strict=True 
    )
    
    lit_model.eval() 

    # ----------------------------------------------------
    # 4. Predict
    # ----------------------------------------------------
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=False
    )
    predictions = trainer.predict(lit_model, test_loader)

    # ----------------------------------------------------
    # 5. Post-Processing & Save
    # ----------------------------------------------------
    all_preds = []
    all_ids = []

    # Processando os batches retornados
    for batch_output in predictions:
        logits, ids = batch_output["logits"], batch_output["ids"]
        probs = torch.sigmoid(logits)

        preds = (probs > 0.5).int()
        
        all_preds.append(preds.cpu().numpy())
        all_ids.extend(ids)

    # Concatenar todos os arrays numpy
    all_preds = np.concatenate(all_preds, axis=0)

    # Criar DataFrame
    df = pd.DataFrame({
        "ID": all_ids,
        "TARGET": all_preds.flatten()
    })
    
    # Salvar
    df.to_csv(args.output_csv, index=False)
    print(f"Resultados salvos em {args.output_csv}")

if __name__ == "__main__":
    main()