import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import os
from ultralytics import YOLO
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG

ROOT = Path(__file__).parent.resolve()


def benchmark_performance(model_path, config) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 30"
    command += f' -shape "[1,3,{config.imgsz},{config.imgsz}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec

    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))



def benchmark(weights_dir, dataset_dir):
    MODEL_NAME = "best"

    images_path = os.path.join(dataset_dir, "test/images")
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    # model = YOLO(f"yolov8s.pt")
    model = YOLO(os.path.join(weights_dir, f"{MODEL_NAME}.pt"))
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = yaml_path

    # Benchmark performance 
    quantized_model_perf = benchmark_performance(os.path.join(weights_dir, f"{MODEL_NAME}.pt"), args)
    print(f"Quantized model performance: {quantized_model_perf} FPS")


def main():
    base_dir = os.path.dirname(os.getcwd())
    models_dir = os.path.join(base_dir, "002-Training-models/train_results/chosen_models")
    dataset_dir = os.path.join(base_dir, "datasets/yolov8_rc")

    for foldername in os.listdir(models_dir):
        variables = foldername.split('_')
        model = variables[1]
        tf = variables[3]
        imgsz = variables[5]
        epochs = variables[7]
        batch = variables[9]
        dataset = variables[11]
        loss = variables[13]
        lr = variables[15]

        print(foldername)
        print("model", model)
        print("tf", tf)
        print("imgsz", imgsz)
        print("epochs", epochs)
        print("batch", batch)
        print("dataset", dataset)
        print("optimizer", loss)
        print("lr", lr)

        weights_dir = os.path.join(models_dir, foldername, "weights")
        print(weights_dir)
        print(dataset_dir)

        benchmark(weights_dir, dataset_dir)


if __name__ == "__main__":
    main()