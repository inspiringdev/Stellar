# STELLAR: LLM-Assisted RSMA Optimisation for LEO Satellite Networks

> Implementation of **"STELLAR: Large Language Model-Assisted Optimization
> for Satellite Networks with RSMA"**, GLOBECOM 2025.

---

## 📋 Overview

STELLAR is a three-phase evolutionary framework that uses a Large Language
Model (LLM) as an intelligent operator to optimise power allocation in
LEO satellite networks with Rate-Splitting Multiple Access (RSMA).

```
Phase 1 – Initialisation : random feasible population
Phase 2 – Evolution      : LLM generates new candidates via few-shot prompting
Phase 3 – Update         : keep top-N individuals by fitness
```

The framework jointly optimises the transmit power of the **common stream**
and each **private stream** to maximise the RSMA sum rate across all ground
stations, subject to total power and minimum per-user rate constraints.

---

## 🗂 Project Structure

```
STELLAR_Implementation/
├── config.py                        ← all simulation hyperparameters
├── main.py                          ← quick single-run demo
├── requirements.txt
├── src/
│   ├── channel.py                   ← LEO Gauss-Markov channel (Eqs 3–5)
│   ├── beamforming.py               ← ZF + MRT initialisation (Eqs 13–14)
│   ├── rsma.py                      ← RSMA / SDMA rate computation (Eqs 6–12)
│   ├── stellar.py                   ← STELLAR algorithm (Algorithm 1)
│   ├── llm_operator.py              ← Anthropic API LLM operator (Fig 1 prompt)
│   └── baselines.py                 ← Random search + Vanilla PPO (PyTorch)
├── experiments/
│   ├── fig2_convergence.py          ← reproduce Fig 2
│   ├── fig3_rate_vs_params.py       ← reproduce Fig 3a & 3b
│   ├── fig4_rsma_vs_sdma.py         ← reproduce Fig 4
│   ├── fig5_population_convergence.py ← reproduce Fig 5
│   └── run_all.py                   ← run all experiments at once
└── results/                         ← output figures + JSON data
```

---

## ⚙️ Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install core dependencies
pip install numpy scipy matplotlib requests

# 3. (Optional) Install PyTorch for the Vanilla PPO baseline
pip install torch
```

---

## 🔑 API Key Setup

STELLAR uses two keys as i wanted to speed up the processing. It uses the **Gemini API** (Google) as well as the **GPT API** (OpenAI) as the LLM operator. You can get your key by going to the websites. I have also initialized the API at the start of the main file as well so you have the option of using the terminal and deleting those two lines or pasting your keys in the main file directly. Below are the commands if you choose option 1:

```bash
export OPENAI_API_KEY="your key"
```

and

```bash
export GEMINI_API_KEY="your key"
```
> **Without the key** the framework can work but the reuslts would be different 
> I have used gemini 2.5 flash and openai 4o mini as the keys for the llm requests  

---

## 🚀 Quick Start

```bash
#use main file to get demo
python main.py

#this is the quick mode
python experiments/run_all.py --quick

#This is the full production
python experiments/run_all.py
```

Output figures are saved to `./results/`.

---

## 📐 System Parameters (matching paper Section IV)

| Parameter | Value |
|---|---|
| Total transmit power Pt | 33 dBm (≈ 2 W) |
| Satellite altitude | 780 km |
| Carrier frequency fc | 4 GHz |
| Satellite antenna gain Gs | 35 dBi |
| GS antenna gain Gk | 24.3 dBi |
| Noise power σ² | −90 dBm |
| Population size N | 15 |
| Parent set size Np | 10 |
| Generations Ng | **50** (Optimized for convergence) |
| Max attempts Em | 5 |
| Primary LLM | `gpt-4o-mini` |
| Fallback LLM | `gemini-2.5-flash` |
| PPO Training Episodes | 30,000 |

---

## 📈 Analysis of Results
Based on the execution of the full suite, the following core findings validate the STELLAR framework:
* **Sample Efficiency:** STELLAR achieves high sum rates (~10.5 bps/Hz) in just **50 generations**. In contrast, standard Random Search requires **50,000 trials** to achieve similar results, proving the LLM is an intelligent optimizer.
* **Performance Gain:** STELLAR demonstrates up to a **54.4% gain** over Vanilla PPO. Traditional Reinforcement Learning struggles with the strict constraints of LEO satellite power allocation, whereas LLMs navigate these constraints effectively.
* **Scalability:** While sum rate naturally drops as more users ($K$) compete for limited satellite power, STELLAR maintains a consistent lead over PPO, with gains reaching as high as **88.3%** for smaller user groups.

---

## 🧮 Key Equations

**Channel model (Eq. 3–5)**
```text
h_k(t) = sqrt(Gs · Gk · (c / 4πfc·dk)²) · g_k(t)
g_k(t) = ρ·g_k(t-1) + sqrt(1-ρ²)·e_k
ρ = J₀(2π·fd·Ts)
```

**Beamforming (Eq. 13–14)**
```text
w̃_c  = Σh_k / |Σh_k|          (MRT, common stream)
w̃_p_k = v_k / |v_k|            (ZF, private streams)
w_c   = sqrt(pc) · w̃_c
w_p_k = sqrt(pk) · w̃_p_k
```

**Fitness (Eq. 17–18)**
```text
f(p_j) = (exp(R_j) - exp(Λ)) / (exp(R_j) + exp(Λ))
```

---

## 📝 Execution Notes

- Due to API rate-limit delays (approx 22s between calls to adhere to free-tier limits), a full execution of `run_all.py` takes roughly **19.5 hours**.
- Averaging over more channel realisations (`N_REALIZ` in `config.py`) smooths the bar charts in Figs 3 & 4.
- The LLM prompt template exactly matches the evolution prompt shown in Fig. 1 of the original paper.

---

## 📜 Citation

```bibtex
@inproceedings{zhang2025stellar,
  title     = {STELLAR: Large Language Model-Assisted Optimization for
               Satellite Networks with RSMA},
  author    = {Zhang, Ruichen and Wang, Jiacheng and Liu, Yinqiu and
               Sun, Geng and Niyato, Dusit and Mao, Shiwen and Sun, Sumei},
  booktitle = {2025 IEEE Global Communications Conference (GLOBECOM)},
  year      = {2025},
}
```
