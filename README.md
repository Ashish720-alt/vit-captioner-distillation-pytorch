## 🏗️ Model Architecture

We train **two components jointly**:

1. **Student Vision Transformer (ViT)** — distilled from CLIP ViT-B/32.  
2. **Causal Decoder** — a Transformer language model that generates captions conditioned on the student embedding.

---

### 🔹 Student ViT (Distilled Image Encoder)

A **tiny Vision Transformer** designed to replicate CLIP’s image embeddings in a lighter, CPU-friendly way.

1. **Patchification**  
   - Input: `224 × 224 × 3` image.  
   - Split into `16 × 16` patches → `(224/16)² = 196` patches.  
   - Each patch flattened to a `768`-dim vector.

2. **Linear Projection**  
   - Each 768-dim patch projected to hidden dim `d = 192`.  
   - Produces a sequence of `196 × 192`.

3. **Class Token**  
   - A learnable `[CLS]` token in `192`-dim is prepended → sequence length = 197.

4. **Positional Embeddings (Learned)**  
   - Unlike fixed sinusoidal encodings, we use **learnable positional vectors**.  
   - A `[197 × 192]` parameter matrix is trained end-to-end and added to the sequence, letting the model learn position-specific representations.  
   - This means positional encodings are optimized during training, not fixed beforehand.

5. **Transformer Encoder**  
   - 6 layers, each with:  
     - Multi-head self-attention (3 heads, 64-dim each).  
     - Feed-forward MLP (hidden size = 768, GELU).  
     - Residual + LayerNorm.

6. **Projection Head**  
   - The `[CLS]` token output `[192]` is linearly mapped to `768`-dim, matching CLIP’s teacher embedding size.

7. **Output**  
   - Normalized 768-dim embedding.  
   - Trained via distillation loss (to CLIP) + captioning loss (via decoder).

---

### 🔹 Causal Decoder (Caption Generator)

A **Transformer decoder** (like a mini language model) that generates captions autoregressively.

1. **Input Embeddings**  
   - Caption tokens → embedded in `d_model = 256`.  
   - Add **learned positional embeddings** (`[max_len × 256]`), which are trainable parameters updated during training.  
   - Ensures the decoder learns optimal ways to represent positions in captions.

2. **Cross-Attention Memory**  
   - Student ViT embedding `[768]` → projected to `[256]`.  
   - This **single vector** acts as the **memory** for all decoder layers.  
   - Inside each decoder layer:  
     - **Query vectors** come from the decoder’s hidden states (caption tokens).  
     - **Key/Value vectors** come from the projected image embedding.  
     - This lets every text token decide how much to attend to the *same* image embedding at each step.  
   - Effectively, the image feature influences every generated token through cross-attention.

3. **Decoder Layers (×2)**  
   - Masked self-attention (causal, prevents peeking ahead).  
   - Cross-attention to the image memory token (Q from text, K/V from image embedding).  
   - Feed-forward MLP (`ffn_dim = 1024`, GELU).  
   - Residual + LayerNorm.

4. **Output**  
   - Decoder hidden states → linear → logits over vocab (~50k).  
   - Trained with autoregressive cross-entropy (predict next token).

---

### 🔹 Training Objective

The total loss:

L = L_caption + λ * L_distill

- **Captioning Loss** \(L_\text{caption}\): Cross-entropy between predicted and true tokens.  
- **Distillation Loss** \(L_\text{distill}\): Cosine distance between student embedding and frozen CLIP teacher embedding.  
- **λ (distill_weight)**: tradeoff factor, set to 1.0.

---
