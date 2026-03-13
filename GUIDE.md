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

Most HAR models (DeepConvLSTM, InnoHAR, etc.) have 200K to 1.5M parameters. They work, but they are too large for on-device deployment on microcontrollers, wearables, or situations where you need fast inference. Our model has ~23,100 parameters and runs in under 10ms. That is the point.

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

**Novel: Learnable Spectral Radius**

In standard reservoir computing, the spectral radius is a fixed hyperparameter. We make it trainable:

```python
init_logit = math.log(spectral_radius / (1.0 - spectral_radius + 1e-7))
self.sr_logit = nn.Parameter(torch.tensor(init_logit))
```

The logit parameterization ensures the spectral radius stays in (0, 1) via sigmoid. During training, gradients flow through the reservoir weight scaling:

```python
sr = torch.sigmoid(self.sr_logit)
W_r = self.W_res * (sr / (self._base_sr + 1e-7))
```

The network learns whether the reservoir should have longer memory (higher SR, closer to 1) or faster forgetting (lower SR). This is the first gradient-based spectral radius optimization in the reservoir computing literature. It adds exactly 1 trainable parameter.

**Novel: Differential Reservoir State Encoding**

Instead of using only the hidden state h(t), we also compute the first-order difference:

```python
h_prev = torch.cat([torch.zeros_like(h_seq[:, :1]), h_seq[:, :-1]], dim=1)
delta_h = h_seq - h_prev
return torch.cat([h_seq, delta_h], dim=-1)
```

delta_h(t) = h(t) - h(t-1) captures the instantaneous rate of change of the reservoir dynamics. For periodic activities like walking, delta_h oscillates with the step frequency. For static activities, delta_h is near zero. This gives the downstream network two complementary views: where the reservoir IS and how fast it is CHANGING.

The two views are merged via a per-neuron learned gate in the main model:

```python
alpha = torch.sigmoid(self.diff_gate)
h = h[:, :, :rs] + alpha * h[:, :, rs:]
```

Each of the 32 reservoir dimensions gets its own mixing weight, adding 32 parameters. The network learns which neurons' derivatives are useful and which are noise.

**Novel: Stochastic Reservoir Masking**

During training, we apply variational dropout to the reservoir states:

```python
if self.training and self.reservoir_dropout > 0:
    drop_mask = (torch.rand(batch, self.reservoir_size, device=x.device) > self.reservoir_dropout).float()
    drop_mask = drop_mask / (1.0 - self.reservoir_dropout)
```

The same mask is used for all 128 timesteps within a sample (variational dropout, Gal and Ghahramani 2016). This is different from standard dropout because the features being masked are non-trainable. The effect: downstream layers cannot rely on any specific reservoir neuron and must learn redundant representations. At inference, the full reservoir is used with no masking.

**Spectral Initialization:**

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

### Stage 3: Spectral Gated Fusion (sensorfusion.py)

This is a core novel contribution. After the DS-Conv processes the reservoir output, we want the network to selectively access raw reservoir dynamics. But instead of a single scalar gate (which treats all frequencies equally), we gate in the frequency domain.

The physical motivation: different human activities have different frequency signatures. Walking produces ~2Hz periodic motion, running ~4Hz, and sitting produces near-zero frequency content. A frequency-domain gate can learn to selectively pass reservoir information at the frequencies that matter while suppressing noise bands.

```python
class SpectralGatedFusion(nn.Module):

    def __init__(self, reservoir_dim, dsconv_channels, seq_len):
        super().__init__()
        self.channel_proj = nn.Conv1d(reservoir_dim, dsconv_channels, kernel_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(seq_len)
        freq_bins = seq_len // 2 + 1
        self.freq_gate = nn.Linear(freq_bins, freq_bins)
        nn.init.zeros_(self.freq_gate.bias)

    def forward(self, reservoir_out, dsconv_out):
        res_projected = self.channel_proj(reservoir_out)
        res_aligned = self.temporal_pool(res_projected)
        dsconv_freq = torch.fft.rfft(dsconv_out, dim=2)
        freq_energy = dsconv_freq.abs().mean(dim=1)
        gate = torch.sigmoid(self.freq_gate(freq_energy))
        res_freq = torch.fft.rfft(res_aligned, dim=2)
        gated_freq = res_freq * gate.unsqueeze(1)
        gated_time = torch.fft.irfft(gated_freq, n=dsconv_out.shape[2], dim=2)
        return dsconv_out + gated_time
```

Step by step:
1. `channel_proj` -- 1x1 conv to project reservoir output (32 channels) to match DS-Conv output (48 channels)
2. `temporal_pool` -- adaptive average pool to match temporal dimension
3. `torch.fft.rfft(dsconv_out, dim=2)` -- compute real FFT of DS-Conv output along time axis. For length 32, this gives 17 complex frequency bins
4. `dsconv_freq.abs().mean(dim=1)` -- compute the average spectral energy across channels, giving a (batch, 17) frequency profile
5. `self.freq_gate(freq_energy)` -- a Linear(17, 17) layer that maps the frequency profile to gate values. Each frequency bin gets its own learned gate
6. `torch.sigmoid(...)` -- squash gate values to [0, 1]
7. `torch.fft.rfft(res_aligned, dim=2)` -- compute FFT of the aligned reservoir signal
8. `res_freq * gate.unsqueeze(1)` -- multiply each frequency bin of the reservoir signal by the corresponding gate value. Gate values near 0 suppress that frequency. Near 1, they pass it through.
9. `torch.fft.irfft(...)` -- inverse FFT to convert back to time domain
10. `dsconv_out + gated_time` -- add the spectrally-gated reservoir residual to the DS-Conv output

The freq_gate bias is initialized to zero, so all gates start at sigmoid(0) = 0.5. During training, the network learns which frequencies of reservoir information help classification. This is fundamentally more expressive than a scalar gate because different activities benefit from different frequency bands of reservoir dynamics.

This adds ~1,890 parameters (1x1 conv + Linear(17, 17) gate).

The old scalar GatedResidualFusion is kept in the codebase for backward compatibility but is not used in the default model or ablation variants.

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

**Novel: Scaled Binary Quantization**

Standard binary networks use raw sign() values, which means every weight has magnitude 1. But some output classes might need stronger or weaker responses. We add per-channel learned scaling:

```python
self.scale = nn.Parameter(torch.ones(out_features))

def forward(self, x):
    w = self.linear.weight
    binary_w = w + (torch.sign(w) - w).detach()
    scaled_w = binary_w * self.scale.unsqueeze(1)
    return nn.functional.linear(x, scaled_w, self.linear.bias)
```

Each output neuron gets its own magnitude factor. The binary weight provides direction (+1/-1), and the scale provides learned magnitude. This is inspired by XNOR-Net's per-filter scaling but applied specifically to the classification head with STE training. It adds only `num_classes` parameters (6 for UCI-HAR, 12 for PAMAP2).

The `export_binary` method extracts actual bit-packed binary weights for deployment:

```python
def export_binary(self):
    signs = torch.sign(self.linear.weight)
    packed = (signs > 0).to(torch.uint8)
    return {"packed_weights": packed, "scale": self.scale, "bias": self.linear.bias}
```

This demonstrates that the binary weights can be stored as single bits, achieving real memory savings beyond the STE training trick.

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
        self.reservoir = EchoStateNetwork(
            input_channels, reservoir_size,
            learnable_sr=True, use_diff_states=True, reservoir_dropout=0.1
        )
        if self.reservoir.use_diff_states:
            self.diff_gate = nn.Parameter(torch.zeros(reservoir_size))
        else:
            self.diff_gate = None
        self.dsconv = DSConvEncoder(in_channels=reservoir_size)
        self.gate = SpectralGatedFusion(
            reservoir_dim=reservoir_size, dsconv_channels=48, seq_len=32
        )
        self.attention = PatchMicroAttention(in_channels=48, seq_len=32, d_model=32, ff_dim=48)
        self.classifier = BinaryClassifier(in_features=32, num_classes=num_classes)

    def forward(self, x, return_aux=False):
        h = self.reservoir(x)             # (batch, 128, 6)  -> (batch, 128, 64)
        h = self._merge_reservoir_states(h)  # (batch, 128, 64) -> (batch, 128, 32)
        h = h.transpose(1, 2)             # (batch, 128, 32) -> (batch, 32, 128)
        dsconv_out = self.dsconv(h)        # (batch, 32, 128) -> (batch, 48, 32)
        x = self.gate(h, dsconv_out)       # spectral gated fusion
        if return_aux:
            x, attn_weights, attn_entropy = self.attention(x, return_attention=True)
        else:
            x = self.attention(x)         # (batch, 48, 32)  -> (batch, 32)
        x = self.classifier(x)            # (batch, 32)      -> (batch, num_classes)
        if return_aux:
            return x, {"attention_entropy": attn_entropy,
                       "attention_weights": attn_weights,
                       "spectral_radius": self.reservoir.effective_spectral_radius}
        return x
```

The `_merge_reservoir_states` method combines the raw states and their derivatives using the learned per-neuron diff_gate. The transpose is needed because Conv1d expects `(batch, channels, length)`.

The `return_aux=True` option returns attention weights, attention entropy, and the current spectral radius alongside the output, enabling entropy regularization during training.

The model also exposes `reservoir_states(x)` and `forward_from_reservoir(h)` methods for reservoir manifold mixup during training.

Parameter count:
- ESN: 1 trainable (spectral radius logit) + 1,217 buffer parameters
- Diff gate: 32 trainable
- DS-Conv: ~7,000 trainable
- Spectral Gate: ~1,890 trainable
- Attention: ~13,900 trainable
- Classifier: ~270 trainable
- **Total: ~23,100 trainable parameters**

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

## Part 4b: Novel Training Techniques

### Reservoir Manifold Mixup (mixup.py)

Standard Mixup (Zhang et al. 2018) interpolates raw inputs: x_mixed = lam * x1 + (1-lam) * x2. Manifold Mixup (Verma et al. 2019) applies this at a hidden layer. Our variant specifically targets the reservoir output:

```python
def reservoir_manifold_mixup(model, x1, x2, y1, y2, criterion, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    h1 = model.reservoir_states(x1)
    h2 = model.reservoir_states(x2)
    h_mixed = lam * h1 + (1.0 - lam) * h2
    output = model.forward_from_reservoir(h_mixed)
    return lam * criterion(output, y1) + (1.0 - lam) * criterion(output, y2)
```

What makes this different from standard Manifold Mixup:

1. The mixing happens in a **non-trainable** feature space. The reservoir defines a fixed nonlinear kernel, so the interpolated representations lie on the reservoir's manifold, not an arbitrary hidden space.
2. The reservoir maps sensor data through tanh nonlinearity with recurrent dynamics. Interpolating in this space mixes temporal dynamics rather than raw amplitude values.
3. Gradients from the mixup loss still flow through the learnable spectral radius (sr_logit), so the reservoir's dynamics adapt even though its weights are frozen.

The mixup is applied with probability 0.3 during training (controlled by --mixup_prob). The loss is weighted at 0.5x the main loss to prevent it from dominating.

### Attention Entropy Regularization

Multi-head attention can collapse -- all heads attend to the same patches, wasting capacity. We add an entropy regularization term to the training loss:

```python
attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
loss = criterion(output, y) - entropy_weight * attn_entropy
```

The entropy is computed per-head: `attn_weights` has shape (batch, num_heads, 8, 8). Higher entropy means more uniform attention (each patch attends to many patches). Lower entropy means concentrated attention (each patch attends to just one or two others).

By subtracting `entropy_weight * entropy` from the loss (note: entropy is positive, so subtracting it penalizes LOW entropy), we encourage the attention to spread across patches. The default entropy_weight is 0.01, providing gentle regularization without forcing completely uniform attention.

This prevents the "attention collapse" problem where one head dominates and others become redundant, which is particularly important with only 2 heads in a lightweight model.

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

Our model: 765,729 MACs, 0.0035 mJ per inference at FP32, 0.0002 mJ at INT8.

---

## Part 13: The Training Pipeline (train.py)

Training uses AdamW optimizer with cosine annealing learning rate schedule.

**AdamW** is Adam with decoupled weight decay. Regular Adam applies weight decay to the gradient, which interferes with the adaptive learning rate. AdamW applies it separately. Weight decay=1e-4 provides mild L2 regularization.

**Cosine annealing** smoothly decreases the learning rate from lr_initial to 0 following a cosine curve over T_max epochs. This is better than step decay because it avoids sharp drops.

The training loop:
1. Load dataset, compute normalization stats (per-channel mean and std from training set)
2. Normalize both train and test sets using training statistics
3. Train for 100 epochs with all novel training features:
   - Each batch calls `model(X, return_aux=True)` to get attention entropy and spectral radius
   - **Attention entropy regularization**: `loss = loss - entropy_weight * aux["attention_entropy"]` (default weight 0.01) prevents attention collapse
   - **Reservoir manifold mixup**: with probability 0.3, interpolates in frozen reservoir state space and adds 0.5x mixed loss
   - **Stochastic reservoir masking**: variational dropout (p=0.1) on reservoir states during training
   - **Learnable spectral radius**: gradient flows through `sr_logit` to optimize reservoir dynamics
4. Save checkpoint with highest test accuracy (stores: model weights, epoch, accuracy, F1 score, normalization stats, dataset name, num classes, activity labels)

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

## Part 15b: How to Know If Your Results Are Good

After running the notebook, you will see many numbers. Here is how to judge whether your model is performing well, and what the key metrics actually mean.

### Overall Accuracy and F1 Score

| Metric | Good | Great | Suspicious |
|--------|------|-------|------------|
| UCI-HAR Accuracy | 90-93% | 93-96% | >98% (likely overfitting or data leak) |
| UCI-HAR F1 Macro | 89-92% | 92-95% | >97% |
| PAMAP2 Accuracy | 85-90% | 90-94% | >96% |
| LOSO Mean Accuracy | 85-90% | 90-93% | >95% |

**Why F1 Macro matters more than accuracy:** Accuracy can be misleading when classes are imbalanced. F1 Macro gives equal weight to every class, so a model that is bad at one rare activity cannot hide behind high accuracy on common activities.

**What to watch for:**
- If accuracy is much higher than F1 Macro (gap > 5%), your model is weak on one or more classes. Check the confusion matrix.
- If training loss keeps dropping but test accuracy plateaus, you are overfitting. Reduce epochs or increase augmentation.
- If test accuracy oscillates wildly across epochs, your learning rate is too high.

### Confusion Matrix

Look at the confusion matrix heatmap. A good model has a bright diagonal and dark off-diagonal cells. Common failure patterns in HAR:

- **Sitting vs Standing confusion:** These activities produce very similar sensor patterns (phone is stationary). A 5-15% confusion rate between them is normal. If it is worse than 15%, the model is underfitting.
- **Walking Upstairs vs Walking Downstairs:** These differ mainly in vertical acceleration trends. Some confusion (5-10%) is expected.
- **Laying misclassified as Sitting/Standing:** This usually means normalization is wrong, since laying has a very distinct gravity axis signature.

### Per-Class F1 Score

| Class | Expected F1 | Why |
|-------|-------------|-----|
| Walking | 0.95+ | Very distinct periodic pattern |
| Upstairs | 0.88-0.94 | Similar to walking but with vertical component |
| Downstairs | 0.88-0.94 | Similar to walking, mirrors upstairs |
| Sitting | 0.85-0.93 | Low motion, easily confused with standing |
| Standing | 0.85-0.93 | Low motion, easily confused with sitting |
| Laying | 0.95+ | Very distinct gravity axis orientation |

If any class is below 0.80, something is wrong with that specific class.

### Ablation Study

The ablation tells you which components contribute the most. Here is how to read it:

- **Full Model should be the best or very close to the best.** If a variant without a component beats the full model, that component may be hurting.
- **No Reservoir** should drop by 2-5%. If it drops more, the reservoir is very important. If it does not drop, the CNN is doing most of the work.
- **No Attention** should drop by 1-3%. Attention helps but is not the biggest contributor for sensor data.
- **No Gate** should drop by 1-4%. If removing the spectral gate has no effect, the fusion is not learning useful frequency-selective behavior.
- **No BinaryHead** may slightly improve accuracy (binary quantization trades some accuracy for compression). A drop of 0.5-1.5% is expected and acceptable because the binary head gives you 8x compression.
- **No DSConv** should drop significantly (3-8%). The CNN is the backbone feature extractor.

### Spectral Radius Learning

The spectral radius starts near 0.9 and should change during training. What to look for:

- **Increases toward 1.0:** The model wants longer memory (good for activities with longer temporal patterns).
- **Decreases toward 0.5-0.7:** The model prefers shorter memory (faster-changing patterns dominate).
- **Stays near 0.9:** The default was already optimal; not much to learn.
- **Crosses 1.0:** This is a red flag. Spectral radius > 1 means the reservoir is unstable (exploding states). The sigmoid parameterization should prevent this, but check if it happens.

### Training Loss Curve

- **Smooth, monotonically decreasing:** Normal training.
- **Spiky but overall decreasing:** The mixup and augmentation are adding stochasticity, which is fine.
- **Flat from the start:** Learning rate is too low or the model is not learning.
- **Drops fast then plateaus early (epoch 10-20):** Learning rate is too high, model converged to a local minimum. Try a smaller LR.
- **Goes up then down:** The entropy regularization and mixup can cause early instability. This is normal if it settles by epoch 20-30.

### Pre-Training Comparison (SimCLR / MSM)

- **SimCLR and MSM should be within +/- 2% of supervised.** They are most useful when you have very little labeled data, which is not the case for UCI-HAR.
- **If SimCLR beats supervised by > 2%:** Your supervised training may be underfitting. Try more epochs or a different LR.
- **If MSM is much worse than supervised:** The mask ratio (15%) may need tuning, or the reconstruction task is too easy/hard.

### Multi-Task Adversarial Training

- **Should be within +/- 1.5% of supervised.** The adversarial head strips subject identity, which helps generalization but can hurt if subjects have very different activity patterns.
- **If much worse:** The adversarial weight (lambda) is too strong. The gradient reversal is destroying activity-relevant information.

### Curriculum Learning

- **Should match or slightly beat standard training.** The curriculum introduces easy activities first (walking, laying) and hard ones later (sitting vs standing).
- **If worse:** The difficulty ordering may not match the actual dataset difficulty. Check if the phase transitions (vertical lines in the plot) align with accuracy jumps.

### LOSO Cross-Validation

- **Standard deviation < 8% across subjects is good.** HAR is inherently subject-dependent.
- **One or two subjects with very low accuracy (< 70%)** is common -- some people walk or sit in unusual ways.
- **Mean LOSO accuracy is the most honest metric.** It tells you how the model performs on truly unseen subjects. Expect it to be 3-8% lower than standard train/test split accuracy.

### Few-Shot Personalization

- **k=5 should improve over base model by 1-5%.** Even 5 examples per class help the model adapt to a new user.
- **k=50 should be the best.** If k=50 is worse than k=20, something is wrong (overfitting on support set).
- **High variance across subjects is expected.** Some subjects are easy to personalize, others are not.

### Adversarial Robustness

- **At epsilon=0 (no attack), accuracy should match clean test accuracy.**
- **At epsilon=0.01-0.02, accuracy should drop by < 5%.** This is mild noise.
- **At epsilon=0.1-0.2, accuracy will drop 20-50%.** This is expected for any model.
- **PGD should be worse than FGSM** at every epsilon. PGD is a stronger attack (iterative).
- **If the model is robust to epsilon=0.5, it has learned meaningful features** rather than relying on fragile input patterns.

### Sensor Drift

- **Accuracy should degrade gracefully.** A sudden cliff means the model relies on absolute sensor values rather than relative patterns.
- **Bias drift is the most damaging** (shifts the mean), followed by scale drift, then noise drift.
- **If accuracy holds up to bias_drift=0.5:** The normalization is working well.

### Activity Transition Detection

- **Stable accuracy should be higher than transition accuracy.** Transitions are inherently harder because the sensor window contains parts of two different activities.
- **Transition accuracy > 60% is decent.** > 70% is good.
- **If transition accuracy is below 40%:** The model is too confident about its predictions and does not handle mixed-activity windows.

### Energy Estimation

- **Total MACs < 1M is excellent** for edge deployment.
- **FP32 energy should be ~4x INT8 energy.** If the ratio is much different, check the MAC estimation.
- **The full model should have slightly more MACs than the ablation variants.** If not, a component is not contributing compute-proportional value.

### Confidence Calibration (ECE)

- **ECE < 0.05 is well-calibrated.** The model's confidence matches its actual accuracy.
- **ECE 0.05-0.10 is acceptable.**
- **ECE > 0.15 means the model is overconfident** (predicts 95% confidence but is only right 80% of the time). Temperature scaling can help.

### Inference Benchmark

- **Mean latency < 10ms on CPU:** The model is real-time capable.
- **P95 < 20ms:** No significant outlier latencies.
- **If GPU latency is higher than CPU for this model:** The model is too small to benefit from GPU parallelism. The overhead of CPU-to-GPU transfer dominates. This is expected and normal for a 23K-param model.

### Cross-Dataset Transfer (UCI-HAR to PAMAP2)

- **Transfer should beat scratch by 2-5% in early epochs.** This shows the pretrained features are useful.
- **By epoch 40-50, scratch may catch up.** With enough training, the randomly initialized model learns PAMAP2-specific features.
- **If transfer is worse than scratch:** The two datasets have incompatible feature distributions. This can happen since PAMAP2 uses a different sensor placement and more activities.

### Noise Robustness

- **Accuracy at SNR=40dB should equal clean accuracy.** 40dB noise is negligible.
- **Accuracy at SNR=10dB should be > 70%.** This is moderate noise.
- **Accuracy at SNR=0dB should be > 40%.** At 0dB, signal and noise have equal power.
- **If accuracy is still high at very low SNR (e.g., 85% at 0dB):** The model may not be using the actual sensor signal. Something is wrong.

---

## Part 16: How to Explain This in a Paper

**Abstract structure:** We present SensorFusion-HAR, a 23.1K-parameter model that introduces learnable spectral radius for Echo State Networks, differential reservoir state encoding, spectral-domain gated fusion, and scaled binary quantization, combined with depthwise separable convolutions and patch micro-attention for real-time HAR. We propose three novel training techniques -- reservoir manifold mixup, attention entropy regularization, and stochastic reservoir masking -- that add zero inference cost. We evaluate on UCI-HAR and PAMAP2 with comprehensive robustness metrics.

**What makes this publishable:**
1. Learnable spectral radius is the first gradient-based spectral radius optimization in reservoir computing, replacing a hyperparameter with a single trainable scalar
2. Spectral-domain gated fusion selectively passes reservoir information at specific frequencies, providing 17-bin frequency resolution where prior work uses a single scalar gate
3. Differential reservoir state encoding captures both the reservoir state and its instantaneous rate-of-change, providing two complementary temporal views from one non-trainable representation
4. Reservoir manifold mixup performs data augmentation in a frozen non-trainable feature space, which is theoretically distinct from standard manifold mixup in trainable layers
5. All 7 contributions add only 296 parameters total, keeping the model under 24K parameters and sub-10ms inference
6. The evaluation suite goes beyond standard metrics to include adversarial robustness, sensor drift tolerance, and activity transition accuracy

**Related work to cite:**
- Echo State Networks: Jaeger (2001), Lukosevicius & Jaeger (2009)
- Depthwise Separable Convolutions: Howard et al. (2017) -- MobileNet
- Self-Attention: Vaswani et al. (2017) -- Transformer
- Binary Neural Networks: Courbariaux et al. (2016) -- BNN, Rastegari et al. (2016) -- XNOR-Net
- SimCLR: Chen et al. (2020)
- BERT / Masked Modeling: Devlin et al. (2019)
- Domain Adversarial Networks: Ganin et al. (2016) -- DANN
- Curriculum Learning: Bengio et al. (2009)
- Manifold Mixup: Verma et al. (2019)
- Variational Dropout: Gal & Ghahramani (2016)
- HAR baselines: Ordonez & Roggen (2016) -- DeepConvLSTM, Xu et al. (2019) -- InnoHAR
- Energy estimation: Horowitz (2014) -- Energy costs of computation
- Spectral analysis in RNNs: Vogt et al. (2007), Verstraeten et al. (2010)

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

### Run Notebook (Google Colab)
Open `sensorfusion_har.ipynb` in Google Colab. The first two code cells automatically install all dependencies, clone the repo, and configure the Python path. No manual setup required -- just click "Run All".

### Run Notebook (Local)
If running locally, open the notebook from inside the `sensorfusion-har` directory. The setup cells detect the local environment and skip cloning.

---

## Part 18: File Map

| File | What It Does |
|------|-------------|
| model/__init__.py | Public API -- re-exports all classes and functions |
| model/reservoir.py | Echo State Network + learnable spectral radius + differential states + stochastic masking |
| model/dsconv.py | Depthwise separable 1D convolutions |
| model/attention.py | Patch micro-attention + entropy regularization |
| model/binary_head.py | Scaled binary quantized classifier (STE) + binary export |
| model/sensorfusion.py | Full pipeline + SpectralGatedFusion + GatedResidualFusion (legacy) |
| model/mixup.py | Reservoir manifold mixup |
| model/augmentation.py | 7-method sensor data augmentation |
| model/contrastive.py | SimCLR contrastive pre-training |
| model/masked_pretrain.py | Masked Sensor Modeling |
| model/multitask.py | Multi-task adversarial training with gradient reversal |
| model/curriculum.py | Curriculum learning scheduler + entropy-aware trainer |
| model/personalization.py | Few-shot personalization |
| model/adversarial.py | FGSM/PGD robustness testing |
| model/transitions.py | Activity transition detection |
| model/drift.py | Sensor drift simulation |
| model/energy.py | MAC-based energy estimation |
| model/visualize.py | t-SNE, attention maps, calibration |
| model/dataset.py | UCI-HAR dataset loader |
| model/dataset_pamap2.py | PAMAP2 dataset loader |
| train.py | Training with entropy reg + manifold mixup |
| evaluate.py | Evaluation + ablation (with proper architecture matching) + benchmark |
| server.py | FastAPI WebSocket server |
| sensorfusion_har.ipynb | 68-cell notebook covering all analysis and experiments |
