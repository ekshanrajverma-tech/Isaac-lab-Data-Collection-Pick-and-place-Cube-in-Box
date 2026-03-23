# Isaac-lab-Data-Collection-Pick-and-place-Cube-in-Box
Repo for steps involved to collect camera and state data from our custom pick and place cube in Box task

# Franka Place Cube Into Box — Isaac Lab Mimic Pipeline

A data collection and augmentation pipeline for training visuomotor policies on a Franka robot placing a cube into a box, built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Task
The Franka robot must pick up a cube from a randomized position on a table and place it inside a box at a randomized position. The task uses a Franka Panda robot with differential IK control and RGB cameras (wrist + table).

## Pipeline Overview
1. **Teleoperation** — Collect ~10 human demos using Isaac Lab record.py
2. **Annotation** — Auto-annotate subtask boundaries (grasp, place)
3. **Generation** — Scale to 200+ demos using Isaac Lab Mimic + CuroBO skillgen

## Dataset
The generated dataset is available on Hugging Face: [YOUR_HF_LINK]

---

## Prerequisites
- Isaac Sim 4.5+
- IsaacLab installed at `~/IsaacLab` — follow the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab)
- CuroBO installed (required for skillgen) — included in IsaacLab extras
- Conda environment `env_isaaclab` set up as part of IsaacLab installation

> **All commands below must be run inside the `env_isaaclab` conda environment:**
> ```bash
> conda activate env_isaaclab
> ```

---

## Setup

### 1. Clone this repo
```bash
conda activate env_isaaclab
git clone https://github.com/YOUR_USERNAME/franka-place-cube-mimic.git
cd franka-place-cube-mimic
```

### 2. Copy env files into IsaacLab
This repo contains two custom environment files that must be placed inside your IsaacLab installation:

| File | Destination |
|------|-------------|
| `franka_place_mimic_env.py` | `~/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/` |
| `franka_place_mimic_env_cfg.py` | `~/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/` |
```bash
cp franka_place_mimic_env.py \
    ~/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/

cp franka_place_mimic_env_cfg.py \
    ~/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/
```

### 3. Update the path in franka_place_mimic_env_cfg.py
`franka_place_mimic_env_cfg.py` imports `env.py` using a hardcoded path. Update line 4 to point to your cloned repo location:
```python
# franka_place_mimic_env_cfg.py — line 4
import sys
sys.path.insert(0, "/path/to/your/franka-place-cube-mimic")  # update this
```

For example if you cloned to your home directory:
```python
sys.path.insert(0, "/home/YOUR_USERNAME/franka-place-cube-mimic")
```

### 4. Register the environment in IsaacLab
Open `~/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py` and add the following block at the end of the file:
```python
gym.register(
    id="Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0",
    entry_point=f"{__name__}.franka_place_mimic_env:FrankaPlaceCubeIntoBoxMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_place_mimic_env_cfg:FrankaPlaceCubeIntoBoxMimicEnvCfg",
    },
    disable_env_checker=True,
)
```

This registers the task ID `Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0` which is used in all commands below.

### 5. Apply IsaacLab bug fix
There is a bug in IsaacLab's `annotate_demos.py` where `subtask_start_signals` are not converted to tensors. Run this once to fix it:
```bash
python3 - << 'EOF'
path = "/home/YOUR_USERNAME/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"

with open(path, "r") as f:
    content = f.read()

old = """            for signal_name, signal_flags in subtask_start_signal_dict.items():
                if not torch.any(signal_flags):"""

new = """            for signal_name, signal_flags in subtask_start_signal_dict.items():
                signal_flags = torch.tensor(signal_flags, device=env.device)
                if not torch.any(signal_flags):"""

assert old in content, "Pattern not found - check your IsaacLab version"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)
print("Done")
EOF
```

---

## End-to-End Run

### Step 1: Collect Teleoperated Demos
Collect ~10 human demonstrations using SpaceMouse or keyboard teleoperation:
```bash
~/IsaacLab/isaaclab.sh -p \
    ~/IsaacLab/scripts/tools/record_demos.py \
    --task Isaac-Place-Cube-Into-Box-Franka-JointPos-Mimic-v0 \
    --dataset_file ./source_demos.hdf5 \
    --num_demos 10 \
    --enable_cameras
```

This produces `source_demos.hdf5` containing your raw teleoperated demonstrations.

### Step 2: Annotate Demos
Auto-annotate subtask boundaries (grasp and place) in the recorded demos:
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

This produces `annotated_demos.hdf5` with subtask boundary labels added. Verify the output before proceeding:
```bash
python3 -c "
import h5py
with h5py.File('./annotated_demos.hdf5', 'r') as f:
    di = f['data/demo_0/obs/datagen_info']
    print('datagen_info keys:', list(di.keys()))
    print('subtask_start_signals:', list(di['subtask_start_signals'].keys()))
    print('subtask_term_signals:', list(di['subtask_term_signals'].keys()))
    print('eef_pose keys:', list(di['eef_pose'].keys()))
"
```

Expected output:
```
datagen_info keys: ['eef_pose', 'object_pose', 'subtask_start_signals', 'subtask_term_signals', 'target_eef_pose']
subtask_start_signals: ['grasp', 'place']
subtask_term_signals: ['grasp', 'place']
eef_pose keys: ['panda_hand']
```

### Step 3: Generate Dataset with Skillgen
Scale up to hundreds of demos using Isaac Lab Mimic with CuroBO collision-aware motion planning:
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

- Increase `--num_envs` for faster generation on more powerful GPUs
- Increase `--generation_num_trials` for a larger dataset
- Progress is printed as `X/Y (Z%) successful demos generated by mimic`

This produces `mimic_demos.hdf5` — your final training dataset.

---

## File Structure
```
franka-place-cube-mimic/
├── README.md
├── env.py                        # Custom env: scene, observations, rewards, terminations
├── franka_place_mimic_env.py     # Mimic env class — copy to IsaacLab (Setup Step 2)
├── franka_place_mimic_env_cfg.py # Mimic env cfg — copy to IsaacLab (Setup Step 2)
└── requirements.txt
```

### What each file does
- **`env.py`** — Defines the scene (robot, cube, box, cameras), observation space, reward terms, and termination conditions.
- **`franka_place_mimic_env.py`** — Extends `ManagerBasedRLMimicEnv` with task-specific mimic API methods: `get_robot_eef_pose`, `action_to_target_eef_pose`, `target_eef_pose_to_action`, `get_subtask_term_signals`, `get_subtask_start_signals`.
- **`franka_place_mimic_env_cfg.py`** — Combines `env.py` config with `MimicEnvCfg` to define subtask configs, selection strategies, and data generation parameters.

---

## Known Issues & Fixes Applied
- **IsaacLab bug** (`annotate_demos.py` line ~419): `subtask_start_signals` loop missing `torch.tensor()` conversion — fixed in Setup Step 5
- **Env cfg**: `subtask_configs` must be set directly on the mimic env cfg, not on `datagen_config`
- **Device mismatch**: `gripper_action` must be on same device as `pose_action` in `target_eef_pose_to_action` — already fixed in `franka_place_mimic_env.py`

---

## Citation
If you use this pipeline, please cite Isaac Lab:
```bibtex
@software{isaaclab,
  title = {Isaac Lab},
  url = {https://github.com/isaac-sim/IsaacLab},
}
```
