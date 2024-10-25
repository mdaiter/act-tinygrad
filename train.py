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

import argparse
# Start of training code
parser=argparse.ArgumentParser(description="Argument Parser for ACT training on simulated environments")
parser.add_argument("env_name", type=str, choices=['aloha_sim_transfer_cube_human', 'aloha_sim_insertion_human'], default='aloha_sim_insertion_human')
parser.add_argument("--model_starting_point", type=str)
parser.add_argument("--model_start_step_count", type=int)
args=parser.parse_args()
env_name = args.env_name

# Create a directory to store the training checkpoint.
output_directory = Path(f"outputs/train/{env_name}")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 100000
log_freq = 1

# Set up the dataset.
delta_timestamps = {
    "action": [i / 50.0 for i in range(100)],
}
dataset = LeRobotDataset(f'lerobot/{env_name}', delta_timestamps=delta_timestamps)
print(dataset.stats)

cfg = ACTConfig()
policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
policy.reset()

step = 0
if args.model_starting_point:
    if (Path(args.model_starting_point).is_file()):
        state_dict = safe_load(args.model_starting_point)
        load_state_dict(policy, state_dict)
        if args.model_start_step_count:
            step = args.model_start_step_count
policy.model.training = True

if cfg.train_backbone_separately:
    params_not_backbone = [p for n, p in nn.state.get_state_dict(policy).items() if p.requires_grad != False and not n.startswith("model.backbone")]
    params_backbone = [p for n, p in nn.state.get_state_dict(policy).items() if p.requires_grad != False and n.startswith("model.backbone")]
else:
    params_not_backbone = nn.state.get_parameters(policy)

Tensor.manual_seed(1000)

if hasattr(cfg, 'override_dataset_stats'):
    for key, stats_dict in cfg.override_dataset_stats.items():
        for stats_type, listconfig in stats_dict.items():
            # example of stats_type: min, max, mean, std
            print(f'listconfig: {listconfig}')
            dataset.stats[key][stats_type] = torch.tensor(listconfig, dtype=torch.float32)

if cfg.train_backbone_separately == True:
    opt = nn.optim.AdamW(params_not_backbone, lr=1e-5, weight_decay=1e-4)
    opt_backbone = nn.optim.AdamW(params_backbone, lr=1e-5, weight_decay=1e-4)
else:
   opt = nn.optim.AdamW(params_not_backbone, lr=1e-5, weight_decay=1e-4)

@TinyJit
@Tensor.train()
def train_step(
    observation_state: Tensor | None = None, 
    observation_images: Tensor | None = None,
    #observation_environment_state: Tensor | None = None,
    action: Tensor | None = None,
    action_is_pad: Tensor | None = None
) -> dict[str, float]:
    Tensor.training = True
    output_dict = policy(observation_state, observation_images, None, action, action_is_pad)
    loss = output_dict["loss"]
    opt.zero_grad()
    if cfg.train_backbone_separately:
        opt_backbone.zero_grad()
    loss.backward()
    if cfg.train_backbone_separately:
        grad_norm_not_backbone = clip_grad_norm_(params_not_backbone, 10.0)
        grad_norm_backbone = clip_grad_norm_(params_backbone, 10.0)
    else:
        grad_norm_not_backbone = clip_grad_norm_(params_not_backbone, 10.0)
    opt.step()
    if cfg.train_backbone_separately:
        opt_backbone.step()
    return (
        loss.realize(),
        grad_norm_backbone.realize() if cfg.train_backbone_separately else grad_norm_not_backbone,
        grad_norm_not_backbone.realize()
    )

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

done = False
with Tensor.train():
    while not done:
        for batch in dataloader:
            batch = {k: Tensor(v.numpy(), requires_grad=False) for k, v in batch.items()}
            batch = policy.normalize_batch_inputs_and_targets(batch)
            print(f'batch: {batch}')
            info = train_step(
                batch["observation.state"].realize(),
                batch["observation.images"].realize(),
                #batch["observation.environment_state"].realize() if "observation.environment_state" in batch else None,
                batch["action"].realize(),
                batch["action_is_pad"].realize()
            )
            loss = info[0]
            grad_norm_backbone = info[1]
            grad_norm_not_backbone = info[2]
        
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.numpy():.3f}")
                print(f"grad_norm_backbone: {grad_norm_backbone.numpy():.3f}")
                print(f"grad_norm_not_backbone: {grad_norm_not_backbone.numpy():.3f}")
            step += 1

            if step % 5000 == 0:
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
