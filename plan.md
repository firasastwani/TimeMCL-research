PROJECT PLAN: Multi-Granularity Hypothesis Learning (TimesBERT × TimeMCL)

0. Objective

Design and implement a novel model that combines:

* TimesBERT-style multi-granularity representation learning
* TimeMCL-style multi-head hypothesis generation with WTA loss

Goal:

Learn specialized hypotheses across temporal scales that compete to explain future trajectories.

⸻

1. Core Idea

Baseline (to outperform)

* Run TimesBERT independently at different granularities
* Combine outputs (ensemble)

Proposed Model

* Single shared encoder (TimesBERT-like)
* Multi-granularity embeddings
* K competing heads (TimeMCL)
* Winner-Takes-All loss

⸻

2. Architecture Overview

Input Time Series x
    ↓
Multi-Granularity Embedding Layer
    (patch sizes: g1, g2, ..., gM)
    ↓
Shared Transformer Encoder (TimesBERT-style)
    ↓
Granularity-Aware Representations
    (z_g1, z_g2, ..., z_gM)
    ↓
K Hypothesis Heads (TimeMCL)
    h1, h2, ..., hK
    ↓
Predicted futures:
    ŷ_1, ŷ_2, ..., ŷ_K
    ↓
Winner-Takes-All Loss

⸻

3. Implementation Steps

⸻

STEP 1: Set Up Base Repos

Clone repositories:

* TimeMCL:

git clone https://github.com/Victorletzelter/timeMCL

* TimesBERT (if not public, replicate from paper):
    * Use a standard Transformer encoder + patch embedding

⸻

STEP 2: Understand TimeMCL Codebase

Focus on:

* tsExperiments/
* training pipeline (train.sh)
* model definition (multi-head outputs)

Key components:

* Shared backbone
* K heads
* WTA loss

⸻

STEP 3: Replace Backbone with TimesBERT-style Encoder

Current:

h = backbone(x)

Replace with:

z_multi = timesbert_encoder(x)

⸻

STEP 4: Implement Multi-Granularity Embedding

Create new module:

class MultiGranularityEmbedding(nn.Module):
    def __init__(self, patch_sizes):
        ...
    
    def forward(self, x):
        outputs = []
        for g in patch_sizes:
            patches = create_patches(x, g)
            emb = embed(patches)
            outputs.append(emb)
        return outputs

⸻

Design details:

* patch_sizes = [1, 4, 8, 16] (tunable)
* each produces a token sequence
* positional encoding required

⸻

STEP 5: Shared Transformer Encoder

Option A (simpler):

* Concatenate all granularity tokens
* Feed into one encoder

Option B (better):

* Separate encoding per granularity
* then fuse

z_g = Transformer(emb_g)

⸻

STEP 6: Granularity Fusion Layer

z = fuse([z_g1, z_g2, ..., z_gM])

Options:

* Concatenation + linear
* Attention over granularities
* Learnable gating (recommended)

⸻

STEP 7: Modify Multi-Head Structure

Original TimeMCL:

y_k = head_k(h)

New version:

y_k = head_k(z)

⸻

Optional (advanced):

Make heads granularity-aware:

y_k = head_k(z, z_gk)

Each head:

* specializes in certain scales

⸻

STEP 8: Keep WTA Loss (CRITICAL)

Do NOT change core training idea.

loss = min_k Loss(y_k, y_true)

Variants:

* hard WTA
* annealed WTA
* soft WTA

⸻

STEP 9: Add Diversity Regularization (optional but recommended)

Encourage heads to differ:

L_div = sum_{i != j} similarity(y_i, y_j)

Total loss:

L_total = L_WTA + λ * L_div

⸻

STEP 10: Training Setup

Use existing TimeMCL scripts:

bash train.sh seed dataset timemcl K awta

Modify:

* model definition
* input pipeline

⸻

STEP 11: Evaluation

Datasets:

* electricity
* traffic
* solar
* exchange
* crypto

Metrics:

* MSE / MAE
* CRPS (if available)
* Diversity score
* Coverage of modes

⸻

4. Baselines (REQUIRED FOR PAPER)

Implement:

Baseline A:

* Standard TimeMCL (no TimesBERT)

Baseline B:

* Multi-run TimesBERT (independent per granularity)

Baseline C:

* Single-head TimesBERT

⸻

5. Key Experiments

⸻

Experiment 1: Performance

Compare:

* ours vs baselines

⸻

Experiment 2: Specialization

Check:

* do heads learn different patterns?

Visualize:

* cluster outputs per head

⸻

Experiment 3: Ablation

Remove:

* multi-granularity → performance drop?
* WTA → collapse?
* shared encoder → effect?

⸻

Experiment 4: Diversity

Measure:

* pairwise distance between heads
* entropy of predictions

⸻

6. Expected Contributions

⸻

1. Conceptual

Multi-scale modeling → hypothesis competition

⸻

2. Technical

* Joint architecture:
    * multi-granularity + multi-head WTA

⸻

3. Empirical

* Better multimodal forecasting
* More diverse predictions
* Better efficiency than ensembles

⸻

7. Risks & Mitigations

⸻

Risk 1: Heads collapse

Fix:

* diversity loss
* temperature annealing

⸻

Risk 2: No improvement over baseline

Fix:

* increase granularity separation
* enforce specialization

⸻

Risk 3: Too heavy model

Fix:

* reduce encoder size
* share weights

⸻

8. Extensions (Optional)

* Granularity routing (each head chooses scale)
* Mixture-of-experts gating
* Pretraining TimesBERT first

⸻

9. Deliverables

* Working PyTorch model
* Training scripts
* Experiment results
* Paper draft sections:
    * Method
    * Experiments
    * Ablations

⸻

10. One-Line Summary

Build a multi-scale, multi-hypothesis forecasting model where each head specializes in different temporal structures via competitive learning.

⸻