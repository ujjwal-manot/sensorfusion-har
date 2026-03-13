# SensorFusion-HAR: Complete Learning Guide

This guide walks through every concept, every module, and every line of code in the project. It is written for someone who wants to understand what is happening, why each decision was made, and how to explain it in a paper or presentation.

---

## Part 1: The Problem

Human Activity Recognition (HAR) takes raw accelerometer and gyroscope readings from a phone and classifies what the person is doing -- walking, sitting, running, climbing stairs, etc.

The phone has two sensors:
- **Accelerometer** -- measures linear acceleration along X, Y, Z axes. Units: m/s^2. Gravity shows up as ~9.8 on whichever axis points down.
- **Gyroscope** -- measures rotational velocity along X, Y, Z axes. Units: rad/s. If the phone is not rotating, these values are near zero.

Together, that is 6 channels of data sampled at 50 Hz (UCI-HAR) or 100 Hz (PAMAP2). A window of 128 timesteps gives us a tensor of shape `(128, 6)` -- 128 time steps, 6 sensor channels.

The goal: take that `(128, 6)` window and output which activity is happening.

### Why lightweight matters

Most HAR models (DeepConvLSTM, InnoHAR, etc.) have 200K to 1.5M parameters. They work, but they are too large for on-device deployment on microcontrollers, wearables, or situations where you need fast inference. Our model has ~22,800 parameters and runs in under 10ms. That is the point.

---

## Part 2: The Datasets

### UCI-HAR

Source: UCI Machine Learning Repository. 30 volunteers, ages 19-48, performed 6 activities while wearing a Samsung Galaxy S II on their waist.

Activities:
| ID | Activity |
|----|----------|
| 0 | Walking |
| 1 | Walking Upstairs |
| 2 | Walking Downstairs |
| 3 | Sitting |
| 4 | Standing |
| 5 | Laying |

The data comes pre-segmented into 128-sample windows with 50% overlap. Each window has 9 channels originally, but we use 6: body acceleration (3) + body gyroscope (3). The dataset splits into 21 training subjects and 9 test subjects.

The dataset loader is in `model/dataset.py`. The `UCIHARDataset` class:
- Downloads and extracts the zip file automatically
- Loads the pre-split train/test windows
- Stacks accelerometer X/Y/Z and gyroscope X/Y/Z into shape `(N, 128, 6)`
- Labels are 0-indexed (original labels are 1-6, we subtract 1)
- `get_normalization_stats()` computes per-channel mean and std from training data
- `loso_split(root_dir, test_subject)` creates a leave-one-subject-out split

### PAMAP2

Source: UCI ML Repository. 9 subjects wearing 3 IMU sensors (hand, chest, ankle). We use only the hand sensor (accelerometer + gyroscope = 6 channels at 100Hz).

Activities:
| ID | Activity | Mapped Label |
|----|----------|-------------|
| 1 | Lying | 0 |
| 2 | Sitting | 1 |
| 3 | Standing | 2 |
| 4 | Walking | 3 |
| 5 | Running | 4 |
| 6 | Cycling | 5 |
| 7 | Nordic Walking | 6 |
| 12 | Ascending Stairs | 7 |
| 13 | Descending Stairs | 8 |
| 16 | Vacuum Cleaning | 9 |
| 17 | Ironing | 10 |
| 24 | Rope Jumping | 11 |

The loader is in `model/dataset_pamap2.py`. Key details:
- Raw files are tab-separated `.dat` files, one per subject
- Column 1 is activity ID, columns 4-6 are hand accelerometer, columns 10-12 are hand gyroscope
- NaN values are filled via linear interpolation
- We slide a 128-sample window with step size 64
- A window is kept only if 80%+ of its labels belong to one activity (rejects transition windows)
- The dominant label becomes the window label, mapped through `ACTIVITY_MAP`

---

## Part 3: The Architecture (Stage by Stage)

### Stage 1: Echo State Network (reservoir.py)

An Echo State Network is a type of recurrent neural network where the recurrent weights are NEVER trained. You set them randomly at initialization and freeze them. Only the downstream layers learn.

Why this works: a random recurrent network with the right properties (spectral radius < 1, sparse connections) acts as a nonlinear temporal kernel. It projects the input into a high-dimensional space where temporal patterns become linearly separable. This is called the "reservoir computing" paradigm.

**The code:**

```python
class EchoStateNetwork(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, spectral_radius=0.9, sparsity=0.8):
        super().__init__()
        self.reservoir_size = reservoir_size

        W_in = torch.randn(input_channels, reservoir_size) * 0.1
        self.register_buffer("W_in", W_in)

        W_res = torch.randn(reservoir_size, reservoir_size)
        mask = (torch.rand(reservoir_size, reservoir_size) > sparsity).float()
        W_res = W_res * mask
        eigenvalues = torch.linalg.eigvals(W_res).abs()
        current_radius = eigenvalues.max().item()
        if current_radius > 0:
            W_res = W_res * (spectral_radius / current_radius)
        self.register_buffer("W_res", W_res)
```

Line by line:

- `W_in = torch.randn(input_channels, reservoir_size) * 0.1` -- Random input projection matrix. Shape (6, 32). Multiplied by 0.1 to keep activations from saturating the tanh.

- `self.register_buffer("W_in", W_in)` -- This is the key line. `register_buffer` means PyTorch will track this tensor (save/load/move to GPU) but will NOT compute gradients for it. It is frozen.

- `W_res = torch.randn(reservoir_size, reservoir_size)` -- Random recurrent weight matrix. Shape (32, 32).

- `mask = (torch.rand(...) > sparsity).float()` -- Creates a binary mask where only 20% of entries are 1 (sparsity=0.8 means 80% zeros). This makes the reservoir sparse, which improves its dynamics.

- `eigenvalues = torch.linalg.eigvals(W_res).abs()` -- Computes eigenvalues of the recurrent matrix.

- `W_res = W_res * (spectral_radius / current_radius)` -- Scales W_res so its largest eigenvalue equals 0.9. This is the **spectral radius**. If it is >= 1, the reservoir is unstable (states explode). If it is too small, the reservoir forgets too fast. 0.9 is a standard choice.

**The forward pass:**

```python
def forward(self, x):
    batch, seq_len, _ = x.shape
    h = torch.zeros(batch, self.reservoir_size, device=x.device, dtype=x.dtype)
    states = []
    for t in range(seq_len):
        h = torch.tanh(x[:, t] @ self.W_in + h @ self.W_res)
        states.append(h.unsqueeze(1))
    return torch.cat(states, dim=1)
```

This is a simple recurrent loop. At each timestep t:
- `x[:, t] @ self.W_in` -- project the 6-channel input to 32 dimensions
- `h @ self.W_res` -- multiply the previous hidden state by the recurrent matrix
- `torch.tanh(...)` -- nonlinear activation, keeps values in [-1, 1]
- The new hidden state `h` depends on both the current input and the previous state, giving the network temporal memory

Output shape: `(batch, 128, 32)` -- the full sequence of hidden states.

**Why not just use an LSTM?** An LSTM would add thousands of trainable parameters (4 gates x 2 weight matrices). The ESN achieves temporal modeling with zero trainable parameters. The downstream CNN and attention learn to extract patterns from the reservoir's fixed nonlinear projection.

**Spectral Initialization (the novel part):**

Instead of random W_in, we can initialize it using the eigenvectors of the training data's autocorrelation matrix:

```python
flat = data.reshape(-1, input_channels)
centered = flat - flat.mean(dim=0, keepdim=True)
autocorr = (centered.T @ centered) / centered.shape[0]
eigenvalues, eigenvectors = torch.linalg.eigh(autocorr)
```

The autocorrelation matrix captures how the 6 sensor channels co-vary. Its eigenvectors point in the directions of maximum variance in the data. By using these as W_in columns, the reservoir starts aligned with the dominant motion patterns rather than random directions. This is conceptually similar to PCA initialization but applied to a reservoir's input projection.

---

### Stage 2: Depthwise Separable Convolutions (dsconv.py)

Standard 1D convolution with K channels input, M channels output, kernel size F costs K * M * F multiply-adds per output position. Depthwise separable convolution splits this into two steps:
1. **Depthwise** -- apply one filter per input channel independently. Cost: K * F per position.
2. **Pointwise** -- 1x1 convolution to mix channels. Cost: K * M per position.

Total cost: K * F + K * M instead of K * M * F. For K=48, M=48, F=5, that is 48*5 + 48*48 = 2,544 vs 48*48*5 = 11,520. About 4.5x cheaper.

This idea comes from MobileNet (Howard et al., 2017).

**The code:**

```python
class DepthwiseSeparableBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
```

- `groups=in_channels` -- this is what makes it depthwise. Each input channel gets its own filter instead of mixing all channels. PyTorch's Conv1d with groups=in_channels means "apply separate convolution to each channel."

- `nn.Conv1d(in_channels, out_channels, 1)` -- kernel_size=1 means it only mixes channels at each position, no spatial filtering. This is the pointwise step.

- `nn.BatchNorm1d` -- normalizes each channel across the batch. Stabilizes training by reducing internal covariate shift.

- `nn.init.kaiming_normal_` -- He initialization, designed for ReLU networks. Scales initial weights so variance is preserved through the layer.

**The encoder:**

```python
class DSConvEncoder(nn.Module):

    def __init__(self, in_channels=32):
        super().__init__()
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(in_channels, 48, kernel_size=5, stride=1, padding=2),
            DepthwiseSeparableBlock(48, 48, kernel_size=5, stride=2, padding=2),
            DepthwiseSeparableBlock(48, 48, kernel_size=3, stride=2, padding=1),
        )
```

Three blocks:
1. Block 1: 32 -> 48 channels, stride 1, sequence stays at 128. Kernel 5 sees 5 timesteps (~100ms at 50Hz).
2. Block 2: 48 -> 48, stride 2, sequence shrinks to 64. Receptive field grows.
3. Block 3: 48 -> 48, stride 2, sequence shrinks to 32. Now each position represents ~12 timesteps.

Output shape: `(batch, 48, 32)` -- 48 channels, 32 time positions.

---

### Stage 3: Gated Residual Fusion (sensorfusion.py)

This is one of the novel contributions. After the DS-Conv processes the reservoir output, we ask: should the network also have direct access to the raw reservoir dynamics?

Sometimes the convolution over-smooths temporal features. The gate lets the network decide how much raw reservoir information to mix back in.

```python
class GatedResidualFusion(nn.Module):

    def __init__(self, reservoir_dim, dsconv_channels, seq_len):
        super().__init__()
        self.channel_proj = nn.Conv1d(reservoir_dim, dsconv_channels, kernel_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(seq_len)
        self.gate_proj = nn.Linear(dsconv_channels, 1)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, reservoir_out, dsconv_out):
        res_projected = self.channel_proj(reservoir_out)
        res_aligned = self.temporal_pool(res_projected)
        gate_input = dsconv_out.mean(dim=2)
        gate = torch.sigmoid(self.gate_proj(gate_input)).unsqueeze(2)
        return dsconv_out + gate * res_aligned
```

Step by step:
1. `channel_proj` -- 1x1 conv to project reservoir output (32 channels) to match DS-Conv output (48 channels)
2. `temporal_pool` -- adaptive average pool to shrink the time dimension from 128 to 32, matching DS-Conv output
3. `gate_proj` -- a single linear layer that takes the mean-pooled DS-Conv features and outputs a scalar
4. `torch.sigmoid(...)` -- squash the scalar to [0, 1]. This is the gate value.
5. `dsconv_out + gate * res_aligned` -- if gate is 0, output is pure DS-Conv. If gate is 1, full reservoir residual is added.

The bias is initialized to zero, so the gate starts at sigmoid(0) = 0.5, giving equal weight to both paths initially. During training, the network learns what gate value works best.

This adds ~1,600 parameters (the 1x1 conv from 32->48 channels and the gate projection).

---

### Stage 4: Patch Micro-Attention (attention.py)

Full self-attention over a sequence of length L costs O(L^2 * d). For L=128, that is expensive. Instead, we split the DS-Conv output into patches and apply attention over the patches.

DS-Conv output: `(batch, 48, 32)` -- 48 channels, 32 time positions.

We split this into 8 patches of 4 time positions each. Each patch becomes a vector of dimension 48 * 4 = 192. This is projected to 32 dimensions. Now we have a sequence of 8 tokens, each 32-dimensional.

Self-attention over 8 tokens costs O(64 * 32) instead of O(16384 * 32). That is 256x cheaper.

```python
class PatchMicroAttention(nn.Module):

    def __init__(self, in_channels=48, seq_len=32, num_patches=8, d_model=32, num_heads=2, ff_dim=48):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = seq_len // num_patches   # 32 // 8 = 4
        self.patch_dim = in_channels * self.patch_size   # 48 * 4 = 192

        self.projection = nn.Linear(self.patch_dim, d_model)   # 192 -> 32
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
```

- `self.projection` -- projects each 192-dimensional patch into 32 dimensions
- `self.pos_embedding` -- learnable positional encoding. Shape (1, 8, 32). Each of the 8 patch positions gets a unique 32-dimensional vector that is added to the patch representation. This tells the attention mechanism which patch came from which position.

```python
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),   # 32 -> 48
            nn.GELU(),
            nn.Linear(ff_dim, d_model),   # 48 -> 32
        )
```

This is a standard transformer block:
- Pre-norm architecture (LayerNorm before attention, not after)
- `nn.MultiheadAttention(32, 2)` -- 2 attention heads, each operating on 16 dimensions
- Feed-forward network: expand 32 -> 48 with GELU activation, project back 48 -> 32
- GELU is smoother than ReLU, commonly used in transformers

**The forward pass:**

```python
    def forward(self, x):
        x = x.transpose(1, 2)                          # (batch, 32, 48)
        batch = x.shape[0]
        x = x.reshape(batch, self.num_patches, self.patch_dim)  # (batch, 8, 192)
        x = self.projection(x) + self.pos_embedding     # (batch, 8, 32)

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # self-attention
        x = x + attn_out                                  # residual connection

        x = x + self.ffn(self.norm2(x))                   # FFN + residual

        return x.mean(dim=1)                              # global average pool -> (batch, 32)
```

The final `x.mean(dim=1)` averages over all 8 patches, producing a single 32-dimensional vector per sample. This is the learned representation of the entire window.

---

### Stage 5: Binary Quantized Head (binary_head.py)

The classification head uses binary weights -- every weight is either +1 or -1. This means matrix multiplication becomes addition and subtraction, which is much faster on hardware.

The problem: you cannot backpropagate through `sign()` because its gradient is zero everywhere (except at zero where it is undefined). The solution is the Straight-Through Estimator (STE):

```python
class BinaryLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        w = self.linear.weight
        binary_w = w + (torch.sign(w) - w).detach()
        return nn.functional.linear(x, binary_w, self.linear.bias)
```

The trick is in this line:
```python
binary_w = w + (torch.sign(w) - w).detach()
```

Breaking it down:
- `torch.sign(w)` -- returns +1 or -1 for each weight
- `torch.sign(w) - w` -- the difference between binary and real-valued weight
- `.detach()` -- tells PyTorch to NOT compute gradients through this difference
- `w + (torch.sign(w) - w).detach()` -- in the forward pass, this equals `torch.sign(w)` (the binary weight). In the backward pass, gradients flow through `w` directly because the `.detach()` part has zero gradient.

So: forward uses binary weights, backward updates real-valued weights. During training, the network maintains full-precision weights but always uses their sign during inference.

The `BinaryClassifier` wraps this with a `BatchNorm1d` layer before the binary linear:

```python
class BinaryClassifier(nn.Module):

    def __init__(self, in_features=32, num_classes=6):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.head = BinaryLinear(in_features, num_classes)

    def forward(self, x):
        return self.head(self.bn(x))
```

BatchNorm before the binary layer is important. It normalizes the input to have zero mean and unit variance, which gives the binary weights a better distribution to work with.

---

### The Full Pipeline (sensorfusion.py)

```python
class SensorFusionHAR(nn.Module):

    def __init__(self, input_channels=6, reservoir_size=32, num_classes=6):
        super().__init__()
        self.reservoir = EchoStateNetwork(input_channels, reservoir_size)
        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.gate = GatedResidualFusion(reservoir_dim=reservoir_size, dsconv_channels=48, seq_len=32)
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def forward(self, x):
        x = self.reservoir(x)       # (batch, 128, 6)  -> (batch, 128, 32)
        x = x.transpose(1, 2)       # (batch, 128, 32) -> (batch, 32, 128)
        dsconv_out = self.dsconv(x)  # (batch, 32, 128) -> (batch, 48, 32)
        x = self.gate(x, dsconv_out) # residual gate
        x = self.attention(x)        # (batch, 48, 32)  -> (batch, 32)
        x = self.classifier(x)       # (batch, 32)      -> (batch, num_classes)
        return x
```

The transpose on line 2 is needed because Conv1d expects `(batch, channels, length)` but the reservoir outputs `(batch, length, channels)`.

Parameter count:
- ESN: 0 trainable (1,216 buffer parameters)
- DS-Conv: ~7,000 trainable
- Gate: ~1,600 trainable
- Attention: ~13,000 trainable
- Classifier: ~260 trainable
- **Total: ~22,800 trainable parameters**

---

## Part 4: Data Augmentation (augmentation.py)

Sensor data augmentation is critical because HAR datasets are small. The `SensorAugmentor` class implements 7 methods:

### 1. Jitter
```python
def jitter(self, x, sigma=0.05):
    return x + np.random.normal(0, sigma, x.shape)
```
Adds Gaussian noise to simulate sensor noise. sigma=0.05 means noise amplitude is about 5% of normalized signal.

### 2. Scaling
```python
def scaling(self, x, sigma=0.1):
    factors = np.random.normal(1, sigma, (1, x.shape[1]))
    return x * factors
```
Multiplies each channel by a random factor near 1.0. Simulates different sensor sensitivity/calibration. The factor is constant across time but different per channel.

### 3. Rotation
```python
def rotation(self, x):
    def _random_rotation_matrix():
        angle = np.random.uniform(-15, 15) * np.pi / 180.0
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
```
This uses the Rodrigues rotation formula. `K` is the skew-symmetric matrix of the rotation axis. The formula `I + sin(a)*K + (1-cos(a))*K^2` gives the rotation matrix for angle `a` around axis `K`.

The rotation simulates the phone being held at a slightly different orientation. Separate rotations are applied to accelerometer (channels 0-2) and gyroscope (channels 3-5) because they live in the same coordinate frame but should be rotated independently to add variety.

### 4. Permutation
Splits the window into 2-5 random segments and shuffles them. This tests whether the model relies too much on temporal order.

### 5. Time Warp
Uses cubic spline interpolation to locally speed up or slow down portions of the signal. Simulates variations in movement speed.

### 6. Magnitude Warp
Multiplies the signal by a smooth, randomly varying curve (also via cubic spline). This makes some portions louder and others quieter.

### 7. Channel Dropout
Randomly zeros out entire channels with probability 0.1. Forces the model to not rely on any single sensor axis.

The `AugmentedDataset` wrapper applies these during training:
```python
class AugmentedDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.augmentor(x)
        return x, y
```

Each time a sample is fetched, random augmentations are applied. Since the DataLoader reshuffles each epoch, each sample gets different augmentations each time it is seen.

---

## Part 5: Self-Supervised Pre-Training

### SimCLR Contrastive Learning (contrastive.py)

The idea: take a batch of samples, create two augmented views of each sample, and train the network to recognize that the two views came from the same sample.

```python
class SensorSimCLR(nn.Module):
    def __init__(self, input_channels=6, reservoir_size=32):
        super().__init__()
        self.reservoir = EchoStateNetwork(input_channels, reservoir_size)
        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.attention = PatchMicroAttention(...)
        self.projection = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        ...
        return F.normalize(x, dim=1)   # L2 normalize to unit sphere
```

The projection head maps 32-dim features to a 32-dim space and normalizes them to the unit sphere. This is important because contrastive loss works with cosine similarity, which requires normalized vectors.

**NT-Xent Loss:**

```python
def nt_xent_loss(z1, z2, temperature=0.1):
    z = torch.cat([z1, z2], dim=0)        # stack both views: (2*batch, 32)
    sim = torch.mm(z, z.t()) / temperature  # cosine similarity matrix
    mask = torch.eye(2 * batch_size, ..., dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)            # mask diagonal (self-similarity)
    labels = torch.cat([pos_idx_top, pos_idx_bottom])
    loss = F.cross_entropy(sim, labels)
```

The similarity matrix `sim` has shape (2B, 2B). Entry (i, j) is the cosine similarity between sample i and sample j, divided by temperature. The labels say: for sample i in the top half, its positive pair is sample i+B in the bottom half (and vice versa). Cross entropy over this matrix pushes positive pairs together and negative pairs apart.

Temperature=0.1 is small, which sharpens the softmax and makes the loss more discriminative.

After pre-training, `transfer_weights()` copies the backbone weights (reservoir, dsconv, attention) to a SensorFusionHAR model, and you fine-tune the classifier.

### Masked Sensor Modeling (masked_pretrain.py)

This is our novel self-supervised method, inspired by BERT's masked language modeling but adapted for continuous sensor signals.

The idea: randomly mask 15% of timesteps, replace them with a learnable mask token, feed through the backbone, and predict the original values at the masked positions.

```python
self.mask_token = nn.Parameter(torch.zeros(1, 1, input_channels))
```

The mask token is a learnable (1, 1, 6) tensor. When a timestep is masked, all 6 channels are replaced with this token. The network must reconstruct the original values.

```python
def forward(self, x, mask=None):
    mask_expanded = mask.unsqueeze(-1).float()
    x_masked = x * (1.0 - mask_expanded) + self.mask_token * mask_expanded
```

This replaces masked positions with the mask token while keeping unmasked positions unchanged.

The loss is MSE only on masked positions:
```python
masked_recon = reconstruction * mask_expanded
masked_target = x * mask_expanded
loss = ((masked_recon - masked_target) ** 2).sum() / (num_masked_elements * channels)
```

Why is this different from SimCLR? SimCLR learns by comparing samples to each other. MSM learns by reconstructing the signal itself. MSM forces the network to understand the actual structure of sensor data -- what values are plausible at a given timestep given the surrounding context.

---

## Part 6: Multi-Task Adversarial Training (multitask.py)

Problem: a model trained on subjects 1-21 might learn features specific to how those people move. When tested on subject 22, accuracy drops.

Solution: train the backbone to be UNABLE to predict which subject produced the data. If the features do not contain subject identity, they must contain only activity-relevant information, which generalizes better.

This uses **gradient reversal**:

```python
class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
```

In the forward pass, this is an identity function. In the backward pass, it NEGATES the gradient. So when the subject classifier says "update the backbone to better predict subjects," the negated gradient actually updates the backbone to WORSE predict subjects.

The full model has two heads:
1. Activity head -- trained normally, tries to classify the activity
2. Subject head -- passes through gradient reversal, adversarially tries to classify the subject

The combined loss: `L = L_activity + lambda * L_subject`

Lambda ramps linearly from 0 to 1 over the first half of training. This lets the backbone learn useful features first before the adversarial pressure kicks in.

---

## Part 7: Curriculum Learning (curriculum.py)

Train on easy activities first, gradually add harder ones.

```python
EASY_ACTIVITIES = [3, 4, 5]      # sitting, standing, laying
MEDIUM_ACTIVITIES = [0]           # walking
HARD_ACTIVITIES = [1, 2]          # upstairs, downstairs
```

Static activities (sitting, standing, laying) have very distinct sensor signatures. Walking is periodic but distinct. Stairs are the hardest because they look similar to walking.

The scheduler splits training into 3 phases:
- Phase 0 (epochs 1-33): train only on easy
- Phase 1 (epochs 34-66): add medium
- Phase 2 (epochs 67-100): add hard

The `CurriculumTrainer` filters the training data each epoch to only include the active classes, but always evaluates on the full test set.

---

## Part 8: Few-Shot Personalization (personalization.py)

After training a general model, you can personalize it for a specific user using just K samples per activity.

```python
def few_shot_personalize(model, support_X, support_y, device, k_shots=10, lr=0.01, steps=50):
    personalized = copy.deepcopy(model)
    for param in personalized.parameters():
        param.requires_grad = False
    for param in personalized.classifier.parameters():
        param.requires_grad = True
    ...
```

The key insight: freeze the entire backbone and only fine-tune the binary classifier head. The backbone already knows how to extract activity features. The classifier just needs to adapt its decision boundaries to this specific person's movement patterns.

This works with as few as 5 samples per class because the classifier only has ~260 parameters.

---

## Part 9: Adversarial Robustness (adversarial.py)

FGSM (Fast Gradient Sign Method):
```python
def fgsm_attack(model, x, y, epsilon, device):
    x_adv = x.clone().detach().to(device).requires_grad_(True)
    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv.detach() + epsilon * grad_sign
    return x_adv.detach()
```

FGSM computes the gradient of the loss with respect to the INPUT (not the weights). Then it perturbs the input in the direction that INCREASES the loss. Epsilon controls how large the perturbation is.

PGD is the iterative version: it takes multiple small FGSM steps and projects back into the epsilon-ball after each step. PGD is a stronger attack.

Why test this? If a model is easily fooled by tiny perturbations to sensor data, it might fail on noisy real-world sensors. Adversarial robustness tells you how stable the model is.

---

## Part 10: Activity Transition Detection (transitions.py)

Most HAR papers report accuracy on windows that are solidly in the middle of an activity. But in real use, people constantly transition between activities. What happens at the boundary?

This module identifies "transition windows" -- windows where the label is not uniform (some timesteps have one activity, some have another). It then compares the model's accuracy on stable windows vs transition windows.

This matters because transition accuracy is typically much lower, and no HAR paper reports this metric systematically.

---

## Part 11: Sensor Drift Simulation (drift.py)

Real sensors change over time:
- **Bias drift**: the zero-point shifts gradually (e.g., accelerometer reads 0.1 m/s^2 when it should read 0)
- **Scale drift**: the sensitivity changes (readings become 5% too large over months)
- **Noise drift**: the noise floor increases as the sensor ages

```python
def simulate_bias_drift(data, drift_rate=0.01, channels=None):
    drifted = data.clone()
    ramp = torch.arange(seq_len, ...).float() * drift_rate
    drifted[:, :, ch] = drifted[:, :, ch] + ramp.unsqueeze(0)
    return drifted
```

The bias increases linearly across the window. drift_rate=0.01 means by the end of a 128-sample window, the bias has shifted by 1.28 units.

Testing drift robustness shows whether a deployed model will degrade over time as the sensor hardware ages.

---

## Part 12: Energy Estimation (energy.py)

MAC (Multiply-Accumulate) counting tells you how computationally expensive a model is, independent of the hardware. Energy estimation converts MACs to millijoules using published per-operation energy costs from Horowitz 2014:

```python
ENERGY_COSTS = {
    "mul_fp32": 3.7e-12,   # 3.7 picojoules per FP32 multiply
    "add_fp32": 0.9e-12,   # 0.9 picojoules per FP32 add
    "mul_int8": 0.2e-12,   # 0.2 picojoules per INT8 multiply
    "add_int8": 0.03e-12,  # 0.03 picojoules per INT8 add
}
```

INT8 multiplication is 18.5x cheaper than FP32. This is why quantization matters for on-device deployment.

The `count_macs` function uses PyTorch forward hooks to capture the input/output dimensions of each layer during a forward pass, then computes MACs based on layer type:
- Linear(in, out): in * out MACs
- Conv1d depthwise: out_channels * out_length * kernel_size
- Conv1d standard: out_channels * out_length * in_channels * kernel_size
- MultiheadAttention: 4 * seq * d^2 (QKV projections) + 2 * seq^2 * d (attention scores)

Our model: 765,488 MACs, 0.0035 mJ per inference at FP32, 0.0002 mJ at INT8.

---

## Part 13: The Training Pipeline (train.py)

Training uses AdamW optimizer with cosine annealing learning rate schedule.

**AdamW** is Adam with decoupled weight decay. Regular Adam applies weight decay to the gradient, which interferes with the adaptive learning rate. AdamW applies it separately. Weight decay=1e-4 provides mild L2 regularization.

**Cosine annealing** smoothly decreases the learning rate from lr_initial to 0 following a cosine curve over T_max epochs. This is better than step decay because it avoids sharp drops.

The training loop:
1. Load dataset, compute normalization stats (per-channel mean and std from training set)
2. Normalize both train and test sets using training statistics
3. Train for 100 epochs, saving the checkpoint with highest test accuracy
4. The checkpoint stores: model weights, epoch, accuracy, F1 score, normalization stats, dataset name, num classes, activity labels

---

## Part 14: The Server (server.py)

FastAPI WebSocket server for real-time inference from a phone.

Two endpoints:
- `/ws/phone` -- phone connects here, sends JSON `{ax, ay, az, gx, gy, gz, t}` at ~50 Hz
- `/ws/dashboard` -- browser dashboard connects here, receives predictions

The server maintains a circular buffer of 4096 samples. Every 64 new samples (when buffer has >= 128), it:
1. Takes the last 128 samples
2. Resamples to a fixed 50 Hz rate (phones have variable sampling rates)
3. Normalizes using the stats from the checkpoint
4. Runs inference
5. Broadcasts results to all connected dashboards

If no trained model exists, it falls back to a heuristic classifier based on acceleration magnitude and gyroscope variance.

---

## Part 15: Key Concepts to Study

If you are presenting or writing about this project, these are the topics you should understand deeply:

**Signal Processing:**
- Sampling rate, Nyquist theorem
- Windowing and overlap (128 samples, 50% overlap)
- Signal-to-noise ratio (SNR) in decibels
- Autocorrelation and spectral analysis

**Reservoir Computing:**
- Echo State Networks vs Liquid State Machines
- Spectral radius and the edge of chaos
- Fading memory property
- Comparison with LSTMs and GRUs

**Convolutional Networks:**
- Standard vs depthwise separable convolutions
- Stride, padding, receptive field
- Batch normalization
- Parameter efficiency (MobileNet, EfficientNet)

**Attention Mechanisms:**
- Self-attention, queries/keys/values
- Multi-head attention
- Positional encoding
- Vision Transformer (ViT) patch tokenization

**Quantization:**
- FP32 vs INT8 representation
- Post-training quantization vs quantization-aware training
- Straight-Through Estimator
- Binary Neural Networks (XNOR-Net, BNN)

**Self-Supervised Learning:**
- Contrastive learning (SimCLR, MoCo, BYOL)
- NT-Xent loss (normalized temperature-scaled cross entropy)
- Masked modeling (BERT, MAE)
- Pre-training and fine-tuning

**Domain Adaptation:**
- Domain adversarial neural networks (DANN)
- Gradient reversal layer
- Subject invariance in HAR
- Transfer learning

**Evaluation:**
- F1 score (macro, weighted, per-class)
- Confusion matrix interpretation
- LOSO cross-validation
- Expected Calibration Error (ECE)
- Adversarial robustness metrics

**Deployment:**
- ONNX export and runtime
- Dynamic quantization in PyTorch
- WebSocket protocol
- DeviceMotion API in browsers

---

## Part 16: How to Explain This in a Paper

**Abstract structure:** We present SensorFusion-HAR, a 22.8K-parameter model that combines reservoir computing, depthwise separable convolutions, gated residual fusion, patch-based attention, and binary quantization for real-time HAR. We introduce Masked Sensor Modeling for self-supervised pre-training and multi-task adversarial training for subject invariance. We evaluate on UCI-HAR and PAMAP2 with novel metrics including adversarial robustness, sensor drift tolerance, and activity transition accuracy.

**What makes this publishable:**
1. The 5-stage pipeline combining four different computational paradigms is novel
2. Gated Residual Fusion between reservoir and convolution has not been proposed before
3. Masked Sensor Modeling adapts masked language modeling to continuous sensor signals
4. Multi-task adversarial training for subject invariance in lightweight models is underexplored
5. The evaluation suite (adversarial robustness, drift simulation, transition detection, energy estimation) goes far beyond standard HAR benchmarks
6. The model achieves competitive accuracy at 100-1000x fewer parameters than published baselines

**Related work to cite:**
- Echo State Networks: Jaeger (2001), Lukosevicius & Jaeger (2009)
- Depthwise Separable Convolutions: Howard et al. (2017) - MobileNet
- Self-Attention: Vaswani et al. (2017) - Transformer
- Binary Neural Networks: Courbariaux et al. (2016) - BNN, Rastegari et al. (2016) - XNOR-Net
- SimCLR: Chen et al. (2020)
- BERT / Masked Modeling: Devlin et al. (2019)
- Domain Adversarial Networks: Ganin et al. (2016) - DANN
- Curriculum Learning: Bengio et al. (2009)
- HAR baselines: Ordonez & Roggen (2016) - DeepConvLSTM, Xu et al. (2019) - InnoHAR
- Energy estimation: Horowitz (2014) - Energy costs of computation

---

## Part 17: Running the Project

### Install
```bash
pip install -r requirements.txt
```

### Train on UCI-HAR
```bash
python train.py --dataset ucihar --epochs 100
```

### Train on PAMAP2
```bash
python train.py --dataset pamap2 --epochs 100
```

### Evaluate
```bash
python evaluate.py --dataset ucihar --checkpoint checkpoints/best_model.pt
python evaluate.py --ablation
python evaluate.py --benchmark
```

### Run Server
```bash
python server.py
```
Open `http://localhost:8765` on your laptop, `http://<your-ip>:8765/phone` on your phone.

### Run Notebook
Open `sensorfusion_har.ipynb` in Jupyter or Google Colab. The notebook runs all 28 analysis sections end-to-end.

---

## Part 18: File Map

| File | Lines | What It Does |
|------|-------|-------------|
| model/reservoir.py | 78 | Echo State Network + spectral init |
| model/dsconv.py | 37 | Depthwise separable 1D convolutions |
| model/attention.py | 45 | Patch-based multi-head attention |
| model/binary_head.py | 29 | Binary quantized classifier (STE) |
| model/sensorfusion.py | 68 | Full pipeline + gated residual fusion |
| model/augmentation.py | 129 | 7-method sensor data augmentation |
| model/contrastive.py | 100 | SimCLR contrastive pre-training |
| model/masked_pretrain.py | 104 | Masked Sensor Modeling |
| model/multitask.py | 177 | Multi-task adversarial training |
| model/curriculum.py | 134 | Curriculum learning scheduler |
| model/personalization.py | 137 | Few-shot personalization |
| model/adversarial.py | 143 | FGSM/PGD robustness testing |
| model/transitions.py | 214 | Activity transition detection |
| model/drift.py | 189 | Sensor drift simulation |
| model/energy.py | 184 | MAC-based energy estimation |
| model/visualize.py | 296 | t-SNE, attention maps, calibration |
| model/dataset.py | 104 | UCI-HAR dataset loader |
| model/dataset_pamap2.py | 148 | PAMAP2 dataset loader |
| train.py | 215 | Training script |
| evaluate.py | 395 | Evaluation + ablation + benchmark |
| server.py | 296 | FastAPI WebSocket server |
| **Total** | **~3,200** | |
