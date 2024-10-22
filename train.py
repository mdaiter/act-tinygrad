from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
import torch

import tinygrad
from tinygrad import Tensor, nn, TinyJit

from omegaconf import ListConfig, OmegaConf

from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

from act import *
from utils import clip_grad_norm_

# Start of training code

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/aloha_sim_insertion_human")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 100000
log_freq = 1

# Set up the dataset.
delta_timestamps = {
    "action": [i / 50.0 for i in range(100)],
}
dataset = LeRobotDataset('lerobot/aloha_sim_insertion_human', delta_timestamps=delta_timestamps)
print(dataset.stats)

cfg = ACTConfig()
policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
policy.reset()

params_not_backbone = [p for n, p in nn.state.get_state_dict(policy).items() if p.requires_grad != False and not n.startswith("model.backbone")]
params_backbone = [p for n, p in nn.state.get_state_dict(policy).items() if p.requires_grad != False and n.startswith("model.backbone")]

Tensor.manual_seed(1000)

if hasattr(cfg, 'override_dataset_stats'):
    for key, stats_dict in cfg.override_dataset_stats.items():
        for stats_type, listconfig in stats_dict.items():
            # example of stats_type: min, max, mean, std
            print(f'listconfig: {listconfig}')
            dataset.stats[key][stats_type] = torch.tensor(listconfig, dtype=torch.float32)

opt = nn.optim.AdamW(params_not_backbone, lr=1e-5, weight_decay=1e-4)
opt_backbone = nn.optim.AdamW(params_backbone, lr=1e-5, weight_decay=1e-4)

#@TinyJit
@Tensor.train()
def train_step(batch):
    Tensor.training = True
    output_dict = policy(batch)
    loss = output_dict["loss"]
    opt.zero_grad()
    opt_backbone.zero_grad()
    loss.backward()
    grad_norm_not_backbone = clip_grad_norm_(params_not_backbone, 10.0)
    grad_norm_backbone = clip_grad_norm_(params_backbone, 10.0)
    opt.step()
    opt_backbone.step()
    info = {
        "loss": loss.item(),
        "grad_norm_backbone": grad_norm_backbone,
        "grad_norm_not_backbone": grad_norm_not_backbone
    }
    return info

print(f'Starting training loop')
# Create dataloader for offline training.
dataloader = DataLoader(
    dataset,
    num_workers=0,
    batch_size=8,
    shuffle=True,
    pin_memory=False,
    drop_last=True,
)

step = 0
done = False
with Tensor.train():
    while not done:
        for batch in dataloader:
            batch = {k: Tensor(v.numpy(), requires_grad=False) for k, v in batch.items()}
            info = train_step(batch)
            loss = info["loss"]
            grad_norm_backbone = info["grad_norm_backbone"]
            grad_norm_not_backbone = info["grad_norm_not_backbone"]
        
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss:.3f}")
                print(f"grad_norm_backbone: {grad_norm_backbone.numpy():.3f}")
                print(f"grad_norm_not_backbone: {grad_norm_not_backbone.numpy():.3f}")
            step += 1

            if step % 4000 == 0:
                try:
                    state_dict = get_state_dict(policy)
                    safe_save(state_dict, f'{output_directory}/model_{step}.safetensors')
                except:
                    print(f'Exception with safe save occured')
            if step >= training_steps:
                done = True
                break

# Save a policy checkpoint.
state_dict = get_state_dict(policy)
safe_save(state_dict, f'{output_directory}/model_final.safetensors')
