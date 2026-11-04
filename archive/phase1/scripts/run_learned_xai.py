import argparse
from xai_metric_behavior.core import load_config
from xai_metric_behavior.runners import run_learned

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/phase1_quick.yaml")
args = p.parse_args()
print(run_learned(load_config(args.config)))
