# Comparative Neuro‑evolution  
## **PPO vs. CMA‑ES on MuJoCo Continuous‑Control Tasks**

> **Tasks:** `Reacher‑v5`, `Ant‑v5`, `HalfCheetah‑v5`, `InvertedDoublePendulum‑v5`  
> **Algorithms:** Proximal Policy Optimization (**PPO**, gradient‑based) **vs.** Covariance Matrix Adaptation Evolution Strategy (**CMA‑ES**, gradient‑free)

---

### Key Take‑aways
* **PPO** converges faster and is more sample‑efficient.
* **CMA‑ES** better explores rugged reward landscapes.
* Both reach competitive final returns across all four tasks.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Layout](#repository-layout)  
3. [Installation & Environment](#installation--environment)  
4. [Reproducing Results](#reproducing-results)  
   * [CMA‑ES](#cma-es)  
   * [PPO](#ppo)  
   * [Running on HPC / Slurm](#running-on-hpc--slurm)  
5. [Results](#results)  
6. [Videos](#videos)  
7. [Plots & Visualization](#plots--visualization)  
8. [Notes & Credits](#notes--credits)

---

## Project Overview
This study benchmarks a policy‑gradient RL method (**PPO**) against a population‑based evolutionary method (**CMA‑ES**) on four MuJoCo continuous‑control environments. We provide:

* **Reproducible code** (Python 3.8)
* **Configurable experiments** via YAML
* **CSV logs** of rewards / fitness
* **Rollout videos** of best policies
* **Automated plots** for learning curves and final‑score distributions

---

## Repository Layout
```text
.
├── README.md                ← this file
├── LICENSE
├── requirements.txt
│
├── code/
│   ├── cma_es_project/      ← CMA‑ES implementation
│   │   ├── config/          ← YAML configs per task
│   │   ├── neural_network/
│   │   ├── optimization/
│   │   ├── training/
│   │   └── utils/
│   └── ppo/                 ← PPO Jupyter notebooks
│
├── scripts/                 ← Slurm job scripts
├── tests/                   ← unit tests
│
├── data/
│   ├── cma_es_runs/         ← CSV logs (CMA‑ES)
│   └── ppo_runs/            ← CSV logs (PPO)
│
├── videos/
│   ├── cma_es/              ← rollout MP4s
│   └── ppo/
│
└── plots/                   ← pre‑generated PNGs
```

---

## Installation & Environment
```bash
# clone the repo
git clone https://github.com/PawanKumarrr/PPO-vs-CMAES-MuJoCo.git
cd PPO-vs-CMAES-MuJoCo

# create & activate virtualenv
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install Python deps
pip install -r requirements.txt
```

### MuJoCo 3.2.6
1. Download MuJoCo 3.2.6 from <https://mujoco.org>.
2. Add environment variables (adjust paths):
   ```bash
   export MUJOCO_PY_MUJOCO_PATH=/path/to/mujoco-3.2.6
   export LD_LIBRARY_PATH=/path/to/mujoco-3.2.6/bin:/path/to/glfw-3.3.8/src:$LD_LIBRARY_PATH
   export MUJOCO_GL=egl   # use 'osmesa' if EGL is unavailable
   ```
3. `gymnasium[mujoco]` is installed through `requirements.txt`.

---

## Reproducing Results
### CMA‑ES
```bash
cd code/cma_es_project

# train (example: Reacher‑v5)
python -m main \
  --mode train \
  --config config/test.yaml \
  --seed 505
```

Config files shipped in `code/cma_es_project/config/`:

| Task | Config | population_size | sigma |
|------|--------|-----------------|-------|
| Reacher‑v5 | `test.yaml` | 50 | 0.3 |
| Ant‑v5 | `test11.yaml` | 50 | 0.1 |
| HalfCheetah‑v5 | `cma_es_2.yaml` | 25 | 0.2 |
| InvertedDoublePendulum‑v5 | `test2.yaml` | 50 | 0.3 |

CSV logs → `data/cma_es_runs/<task>/rewards.csv`

**Evaluate best policy**
```bash
python -m main \
  --mode evaluate \
  --config config/test.yaml \
  --checkpoint logs/checkpoints/best_model.pth \
  --seed 505
```

### PPO
```bash
cd code/ppo
jupyter notebook
```
Run notebooks:
* `Reacher_PPO.ipynb`
* `Ant_PPO.ipynb`
* `Halfcheetah_PPO.ipynb`
* `IDP_PPO.ipynb`

Logs saved to `data/ppo_runs/<task>/…`

*(Optional) convert notebooks to scripts:*
```bash
pip install nbconvert
jupyter nbconvert --to script *.ipynb
python ppo_reacher.py
```

### Running on HPC / Slurm
Adapt `code/scripts/*.sh` to your paths, then:
```bash
sbatch code/scripts/test5.sh
```

---

## Results
| Task | PPO Return ↑ | CMA‑ES Return |
|------|--------------|---------------|
| Reacher‑v5 | **≈ −5** | −15 |
| Ant‑v5 | **≈ 2000** | 1800 |
| HalfCheetah‑v5 | **≈ 5500–6000** | 2500–3000 |
| InvertedDoublePendulum‑v5 | **≈ 9000** | 6000 |

---

## Videos
| Task | PPO | CMA‑ES |
|------|-----|--------|
| Reacher‑v5 | `videos/ppo/reacher.mp4` | `videos/cma_es/reacher.mp4` |
| Ant‑v5 | `videos/ppo/Ant.mp4` | `videos/cma_es/Ant.mp4` |
| HalfCheetah‑v5 | `videos/ppo/HalfCheetah.mp4` | `videos/cma_es/HalfCheetha.mp4` |
| InvertedDoublePendulum‑v5 | `videos/ppo/IDP.mp4` | `videos/cma_es/IDP.mp4` |

> **Large MP4s** are mirrored on Google Drive – see link in the paper.

---

## Plots & Visualization
* Pre‑generated PNGs live in `plots/<task>/`.
* Re‑generate CMA‑ES learning curves:
  ```bash
  python code/cma_es_project/utils/plotter.py --run_dir data/cma_es_runs/Reacher
  ```
* PPO curves are produced inside notebooks.

---

## Notes & Credits
* Large checkpoints (`best_model.pth`, `train_results.npy`) are excluded from version control.
* Experiments used MuJoCo 3.2.6 + Python 3.8.
* Slurm scripts assume an EGL‑enabled cluster; tweak for local runs.

### Citation
```
Author (2025). Comparative Neuro‑evolution: PPO vs. CMA‑ES in MuJoCo Environments. GitHub. https://github.com/PawanKumarrr/PPO-vs-CMAES-MuJoCo
```

Released under the MIT License – see [LICENSE](LICENSE).

---

*Happy experimenting 🚀*

