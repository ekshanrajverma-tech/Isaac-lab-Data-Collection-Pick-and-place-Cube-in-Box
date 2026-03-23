# Isaac-lab-Data-Collection-Pick-and-place-Cube-in-Box
Repo for steps involved to collect camera and state data from our custom pick and place cube in Box task

# Franka Place Cube Into Box — Isaac Lab Mimic Pipeline

A data collection and augmentation pipeline for training visuomotor 
policies on a Franka robot placing a cube into a box, built on 
[Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Task
The Franka robot must pick up a cube from a randomized position 
on a table and place it inside a box at a randomized position.

## Pipeline
1. **Teleoperation** — Collect ~10 human demos using Isaac Lab record.py
2. **Annotation** — Auto-annotate subtask boundaries (grasp, place)
3. **Generation** — Scale to 200+ demos using Isaac Lab Mimic + CuroBO skillgen

## Setup
```bash
# Clone IsaacLab and install
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh -i

# Clone this repo into your workspace
git clone https://github.com/YOUR_USERNAME/franka-place-cube-mimic.git
cd franka-place-cube-mimic

# Register the custom env
# Add to IsaacLab's mimic envs __init__.py (see instructions below)
```

## Step 1: Collect Teleoperated Demos
```bash
~/IsaacLab/isaaclab.sh -p \
    ~/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0 \
    --dataset_file ./source_demos.hdf5 \
    --num_demos 10 \
    --enable_cameras
```

## Step 2: Annotate Demos
```bash
~/IsaacLab/isaaclab.sh -p \
    ~/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0 \
    --input_file ./source_demos.hdf5 \
    --output_file ./annotated_demos.hdf5 \
    --auto \
    --annotate_subtask_start_signals \
    --enable_cameras \
    --headless
```

## Step 3: Generate Dataset with Skillgen
```bash
~/IsaacLab/isaaclab.sh -p \
    ~/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0 \
    --input_file ./annotated_demos.hdf5 \
    --output_file ./mimic_demos.hdf5 \
    --num_envs 4 \
    --generation_num_trials 200 \
    --use_skillgen \
    --enable_cameras \
    --headless
```

## Dataset
The generated dataset is available on Hugging Face:
[YOUR_HF_LINK]

## Known Issues & Fixes Applied
- IsaacLab bug: `subtask_start_signals` missing `torch.tensor()` 
  conversion in `annotate_demos.py` (line ~419)
- Custom env requires `subtask_configs` set on env cfg directly,
  not on `datagen_config`
