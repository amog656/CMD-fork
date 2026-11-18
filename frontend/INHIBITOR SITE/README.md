# GANDIVA / CMD-GEN: Target-Configurable Ensemble-Aware Pharmacophore & Selective Inhibitor Design (V13.3)

**Team G-1591 NGIT | DRUG PARADIGM**  
*Mentor: Dr. T. Swathi*  
*Team Members: A. Amogh · D. Sri Karthikeya · D. Hanut Sai · U. Abhiram Sai · P. S. Sreyash*

---

## 🌟 Executive Summary

GANDIVA (CMD-GEN V13.3) is an advanced structure-based AI platform for rational, target-selective inhibitor design. It overcomes the fundamental limitations of conventional generative models (static crystal structures, single pharmacophores, unconstrained docking triage, and high false-positive rates) through an **ensemble-aware, pharmacophore-guided generative pipeline** guarded by **10 preregistered decision gates**.

---

## 🔬 Core Architectural Pillars (Proposed Model vs. Conventional Base)

As benchmarked in **Image 15** of the project specification:

| Aspect | Base Model (Conventional) | Proposed Model (This Work) | Improvement / Clinical Impact |
| :--- | :--- | :--- | :--- |
| **Protein Modelling** | Single static crystal structure | Multiple protein conformations (**aSAMt ensemble**) | Captures loop flexibility & induced-fit breathing |
| **Pharmacophore Design** | Single rigid pharmacophore | Consensus & differential pharmacophores (**PharmFlow**) | Isolates target-selective vs conserved binding features |
| **Molecular Generation** | Conventional unguided generation | Pharmacophore-guided transformer graph generator (**Frozen GCPG**) | High throughput (~1.2s/mol) & structural compatibility |
| **Pocket Analysis** | Fixed binding pocket | Multiple pocket states & domain routing | Handles conformational variability & cryptic pockets |
| **Molecular Evaluation** | Primarily docking scores | Multi-tiered scoring funnel (**Docking → MM-GBSA → FEP → MD**) | Comprehensive multi-level energetic validation |
| **Quality Control** | Ad-hoc / limited validation | **10-Stage Preregistered Decision Gates** | Prevents false-positive experimental claims |
| **Generalization** | Limited OOD assessment | Out-of-distribution (**OOD**) & retrospective tests | Guarantees transferability across unseen target families |
| **Reproducibility** | Limited replicate testing | Systematic replicate & leakage checks | Robust, statistically defensible hit rates |
| **Target Adaptability** | Rigid, target-specific setup | **Target-Configurable Card (TCC)** framework | Seamlessly configures PARP1, USP1, ATM, etc. |

---

## 🚀 The 10-Stage Generative Pipeline

### Phase 1: Inputs & Setup
- **Stage 0: Inputs & Applicability [Gate G0]**  
  Locks the Target Configuration Card (TCC), validates resolution, construct integrity, and catalytic residues.
- **Stage 1: aSAMt Ensemble Sampling (Frozen)**  
  Generates 5–20 apo/holo-seeded conformers capturing protein loop dynamics and active site breathing.
- **Stage 2: QC & Restrained Relaxation [Gate Q]**  
  Evaluates Ramachandran angles, steric clashes, and verifies pocket RMSD $\le 2.0\text{ \AA}$.
- **Stage 3: Pocket States & Domain Routing [Gate G1]**  
  Clusters conformational states with $\ge 90\%$ reference-state coverage; routes in-domain holo states into Phase 2.

### Phase 2: Pharmacophore Design & ML Generation
- **Stage 4: PharmFlow 3D Pharmacophore Diffusion [Gates G-LFO, G2a, G2b]**  
  SE(3)-equivariant continuous diffusion / flow matching samples 3D pharmacophore point clouds (HBD, HBA, H, AR, POS, NEG).
- **Stage 5: Consensus & Differential Filtering [Gates G3, D-state, G4]**  
  Extracts conserved core features and computes differential selectivity scores:  
  $$S(f) = P_{\text{Target}}(f) - \max P_{\text{Off-Target}}(f)$$
- **Stage 6: Frozen GCPG Molecular Generation**  
  Transformer encoder-decoder maps 3D pharmacophores into 2D/3D chemical graphs (~1.2s per molecule) gated on CMD-GEN (R1: Docking) or CMD-GEN (R2: Physicochemical properties).

### Phase 3: Evaluation, Funnel & Decisions
- **Stage 7: Early Chemistry Filters**  
  Filters candidates against PAINS, reactive Brenk alerts, Lipinski Rule of 5, and synthetic accessibility (SAS).
- **Stage 8: Scoring Funnel & FEP Shortlist [Gate FEP]**  
  Multi-state Vina ensemble docking $\to$ MM-GBSA binding energy $\to$ Free Energy Perturbation (FEP) $\Delta\Delta G$ selectivity.
- **Stage 9: MD Persistence & Wet-Lab Experimental Claims [Gate G7-8]**  
  100 ns MD trajectory assessing key contact persistence ($>90\%$), followed by biochemical $IC_{50}$ assay verification.

---

## 🛡️ The 10 Preregistered Decision Gates

1. **G0 (Inputs & Applicability)**: STOP gate. Missing required construct evidence $\to$ NOT EVALUABLE / exploratory only.
2. **Q (Ensemble QC)**: STOP gate. Requires $\ge 80\%$ conformers passing QC and pocket RMSD $\le 2.0\text{ \AA}$.
3. **G1 (Pocket/State Coverage)**: FALLBACK gate. Requires $\ge 90\%$ reference-state coverage within $1.5\text{ \AA}$.
4. **G-LFO (Family Generalization)**: DOWNGRADE gate. Evaluates $\ge 10$ training families and $\ge 5$ held-out families without leakage.
5. **G2a (PharmFlow vs. DiffPhar)**: DOWNGRADE gate. Holm-adjusted one-sided test for $\Delta F_1 > 0$ superiority.
6. **G2b (Ensemble vs. Crystal)**: FALLBACK gate. Paired ensemble-crystal AUC superiority check.
7. **G3 (Differential-Feature Separability)**: DOWNGRADE gate. Block-aware CV $AUC \ge 0.75$, permutation $p < 0.01$.
8. **D-state (Mode D Specificity)**: DOWNGRADE gate. State ablation on held-out actives showing potency AUC drop $\ge 0.05$.
9. **G4 (Retrospective GO/NO-GO)**: STOP gate. Primary H5 hypothesis requires AUC gain $\ge 0.05$ before unlocking GCPG generation.
10. **FEP Gate**: DOWNGRADE gate. Preregistered 95% CI clearing threshold for $\Delta\Delta G$ selectivity.
11. **G7-8 (Experimental Claims)**: DOWNGRADE gate. Biochemical $IC_{50}$ potency hit and Wilson 95% CI selectivity against off-target.

---

## 💻 Running the Platform

### Option 1: One-Click Startup (Windows)
Double-click `start_server.bat` in the project root. It will start the server and open your default browser to `http://localhost:3000`.

### Option 2: Command Line Startup
```bash
# Start backend server
agy-node server.js
```
Then visit: `http://localhost:3000`

### Option 3: Standalone / Offline Mode
The frontend is completely resilient. You can also double-click `index.html` or `GANDIVA.html` directly in any web browser; the built-in embedded simulation engine will take over automatically!

---

## 📁 Project Directory Structure
```
INHIBITOR SITE/
├── server.js                      # Node.js backend server with REST API & simulation engine
├── index.html                     # Primary modern SPA frontend interface
├── GANDIVA.html                   # Dual-mode standalone & networked application
├── style.css                      # Cyber-biotech responsive dark UI styling
├── app.js                         # Three.js 3D studio, 2D molecule renderer, pipeline controller
├── start_server.bat               # Windows launcher script
├── README.md                      # Complete system documentation
└── data/                          # Pre-cached curated PDB structures
    ├── 7ONS.pdb                   # PARP1 human catalytic domain (primary target)
    ├── 4ZZY.pdb                   # PARP2 catalytic domain (off-target homolog)
    └── 8I63.pdb                   # USP1 catalytic domain
```
