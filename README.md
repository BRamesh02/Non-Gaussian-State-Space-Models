# Exact and Particle Filtering for Non-Gaussian State Space Models

This repository contains the code for an academic project carried out in the context of
the **ENSAE course _Hidden Markov Models and Sequential Monte-Carlo Methods_**.

**Authors**  
- Boyina Leena  
- Ramesh Brian  
- Tonin Ireni  

---

## Project Description

This project is based on the paper:

> **Drew Creal (2012)**  
> *A Class of Non-Gaussian State Space Models with Exact Likelihood Inference*  
> Available at: https://sites.google.com/view/drewcreal/home

The paper proposes an **exact filtering algorithm** for a specific class of non-Gaussian
state-space models.  
Our work focuses on the **first model introduced in Section 5.1** of the paper.

---

## Objectives

The project is structured around three main tasks.

### 1. Filtering on simulated data
- Implement:
  - a **bootstrap particle filter**,
  - the **exact filter** proposed by Creal.
- Compare both approaches on **simulated data** (true parameters are known):
  - likelihood profiles,
  - filtering distributions.
- The comparison follows the spirit of **Figure 1** in the paper.

### 2. Parameter estimation using the exact likelihood
- Implement a **Random-Walk Metropolis–Hastings (RWMH)** algorithm.
- Use the **true likelihood**, computed via the exact filtering method.
- Estimate model parameters for the data considered in the paper (or simulated analogues).

### 3. Particle Marginal Metropolis–Hastings (PMMH)
- Implement a **PMMH algorithm** relying on the bootstrap particle filter.
- Compare its performance with the *ideal* MCMC sampler from Task 2:
  - mixing behavior,
  - autocorrelation functions (ACFs),
  - impact of the number of particles \(N\).
- This analysis follows the discussion from the course on **PMMH calibration**.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── task_1.ipynb           # Exact vs bootstrap filtering comparison
│   ├── task_2.ipynb           # (Optional / intermediate experiments)
│   └── task_3.ipynb           # MCMC and PMMH inference experiments
│
├── src/
│   ├── cox_simulation.py      # Simulation of the state-space model (Section 5.1)
│   ├── creal_filter.py        # Exact filtering algorithm (Creal, 2012)
│   ├── particle_filter.py    # Bootstrap particle filter
│   └── __pycache__/
│
├── utils/
│   ├── plots.py               # Plotting utilities (likelihoods, filters, ACFs)
│   └── __pycache__/
│
├── README.md
└── .gitignore



---

## Main Components

### `cox_simulation.py`
- Simulates the non-Gaussian state-space model from Section 5.1
- Allows full control over true parameter values

### `creal_filter.py`
- Implements the **exact filtering algorithm**
- Computes the **true likelihood** of the observed data

### `particle_filter.py`
- Implements a **bootstrap particle filter**
- Returns filtering distributions and likelihood estimates

### `utils/plots.py`
- Helper functions for:
  - Likelihood profiles
  - Filtering distribution plots
  - Autocorrelation functions (ACFs)
  

---

## References

- Creal, D. (2012). *A Class of Non-Gaussian State Space Models with Exact Likelihood Inference*.
- Andrieu, C., Doucet, A., & Holenstein, R. (2010). *Particle Markov chain Monte Carlo methods*.

---

## Notes

- The exact filter serves as a **benchmark** to assess particle methods.
- PMMH performance is sensitive to the number of particles; this project illustrates
  the trade-off between computational cost and mixing efficiency.

---