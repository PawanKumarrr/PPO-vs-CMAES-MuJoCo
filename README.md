Comparative Neuroevolution: PPO vs. CMA-ES in MuJoCo Environments
This repository contains source code, results, and videos for a comparative study of Proximal Policy Optimization (PPO) and Covariance Matrix Adaptation Evolution Strategy (CMA-ES) in MuJoCo continuous-control tasks: Reacher-v5, Ant-v5, HalfCheetah-v5, and InvertedDoublePendulum-v5.
Project Overview
This project evaluates gradient-based (PPO) and gradient-free (CMA-ES) methods for evolving control policies in MuJoCo environments. Key findings:

PPO excels in sample efficiency and convergence speed.
CMA-ES offers robust exploration in complex reward landscapes.
Experiments cover all four tasks, with results and videos for both methods.

The repository includes Python code, Jupyter notebooks, configuration files, result CSVs, and rollout videos. The full report is maintained separately.
Repository Structure
├── README.md
├── LICENSE
├── requirements.txt
├── code/
│   ├── cma_es_project/     # CMA-ES implementation
│   │   ├── config/         # Configuration YAML files
│   │   ├── neural_network/ # Neural network models
│   │   ├── optimization/   # CMA-ES algorithm
│   │   ├── training/       # Training and evaluation scripts
│   │   ├── utils/          # Logging and plotting utilities
│   ├── ppo/                # PPO Jupyter notebooks
│   ├── scripts/            # Slurm job scripts for CMA-ES
│   ├── tests/              # Unit tests
├── data/
│   ├── cma_es_runs/        # CMA-ES results (rewards.csv)
│   ├── ppo_runs/           # PPO results (CSVs from notebooks)
├── videos/
│   ├── cma_es/             # CMA-ES rollout videos
│   ├── ppo/                # PPO rollout videos
├── plots/                  # Visualization plots

Installation

Clone the Repository:
git clone https://github.com/your-username/PPO-vs-CMAES-MuJoCo.git
cd PPO-vs-CMAES-MuJoCo


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Install MuJoCo:

Download MuJoCo 3.2.6 from mujoco.org.
Set environment variables, replacing /path/to/ with your MuJoCo and GLFW paths:export MUJOCO_PY_MUJOCO_PATH=/path/to/mujoco-3.2.6
export LD_LIBRARY_PATH=/path/to/mujoco-3.2.6/bin:/path/to/glfw-3.3.8/src:$LD_LIBRARY_PATH
export MUJOCO_GL=egl


Install gymnasium[mujoco] via requirements.txt.



Reproducing Results
CMA-ES

Navigate to CMA-ES Code:
cd code/cma_es_project


Run Training:Use configuration files in code/cma_es_project/config/:

Reacher-v5: test.yaml (population_size=50, sigma=0.3)
Ant-v5: test11.yaml (population_size=50, sigma=0.1)
HalfCheetah-v5: cma_es_2.yaml (population_size=25, sigma=0.2)
InvertedDoublePendulum-v5: test2.yaml (population_size=50, sigma=0.3)

python -m main --mode train --config config/test.yaml --seed 505

Results are saved in data/cma_es_runs/<environment>/rewards.csv.

Evaluate Best Model:Use the checkpoint from training (not included in repo due to size):
python -m main --mode evaluate --config config/test.yaml --checkpoint logs/checkpoints/best_model.pth --seed 505


Run on HPC (Optional):Submit Slurm jobs using scripts in code/scripts/ (e.g., test5.sh):
sbatch code/scripts/test5.sh

Edit scripts to update paths for your HPC environment (e.g., MUJOCO_PY_MUJOCO_PATH, PYTHONPATH).


PPO

Run Notebooks:Open notebooks in code/ppo/ using Jupyter:
cd code/ppo
jupyter notebook

Run:

Reacher_PPO.ipynb
Ant_PPO.ipynb
Halfcheetah_PPO.ipynb
IDP_PPO.ipynbResults are saved in data/ppo_runs/<environment>/ (e.g., rewards.csv).


Convert to Scripts (Optional):
pip install nbconvert
jupyter nbconvert --to script *.ipynb
python ppo_reacher.py



Results

Reacher-v5: PPO ~ -5, CMA-ES ~ -15
Ant-v5: PPO ~ 2000, CMA-ES ~ 1800
HalfCheetah-v5: PPO ~ 5500-6000, CMA-ES ~ 2500-3000
InvertedDoublePendulum-v5: PPO ~ 9000, CMA-ES ~ 6000

Results are in data/cma_es_runs/<environment>/rewards.csv for CMA-ES and data/ppo_runs/<environment>/ for PPO.
Videos
Videos showcase the best PPO and CMA-ES policies for each task:

Reacher-v5:
PPO: videos/ppo/reacher.mp4
CMA-ES: videos/cma_es/reacher.mp4


Ant-v5:
PPO: videos/ppo/Ant.mp4
CMA-ES: videos/cma_es/Ant.mp4


HalfCheetah-v5:
PPO: videos/ppo/HalfCheetah.mp4
CMA-ES: videos/cma_es/HalfCheetah.mp4


InvertedDoublePendulum-v5:
PPO: videos/ppo/IDP.mp4
CMA-ES: videos/cma_es/IDP.mp4



Large videos are hosted on Google Drive: Link
Plots
Visualization plots (e.g., learning curves, box plots) are in plots/<environment>/. Regenerate using :

CMA-ES: code/cma_es_project/utils/plotter.py
PPO: Notebooks in code/ppo/

Notes

The report, maintained separately, references this repository for code, results, and videos.
Large files (e.g., best_model.pth, train_results.npy) are excluded to keep the repository lightweight.
Slurm scripts (e.g., test5.sh) are tailored for an HPC environment with MuJoCo 3.2.6 and Python 3.8. Adjust paths for local or other systems.


