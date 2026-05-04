"""Scan notebook for likely undefined names."""
import json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

nb = json.load(open('sensorfusion_har_OPTIMIZED.ipynb', encoding='utf-8'))

# Names that will be available after Cell 3 imports
known = {
    'os','sys','copy','time','random','warnings','np','torch','nn','DataLoader','plt',
    'confusion_matrix','classification_report','f1_score','ConfusionMatrixDisplay',
    'UCIHARDataset','PAMAP2Dataset','MergedHARDataset','MERGED_ACTIVITY_LABELS','build_merged_dataset',
    'SensorFusionHAR','GatedResidualFusion','SpectralGatedFusion',
    'SensorFusionLite','MaskedSensorModelLite','EchoStateNetworkLite','DSConvEncoderLite','SimpleFCGate','transfer_masked_weights_lite',
    'EchoStateNetwork','DepthwiseSeparableBlock','DSConvEncoder','PatchMicroAttention','BinaryLinear','BinaryClassifier',
    'SensorAugmentor','AugmentedDataset',
    'SensorSimCLR','nt_xent_loss','pretrain_contrastive','transfer_weights',
    'MaskedSensorModel','create_mask','masked_pretrain','transfer_masked_weights',
    'GradientReversalLayer','MultiTaskHAR','SubjectLabeledDataset','train_multitask',
    'CurriculumScheduler','CurriculumTrainer',
    'few_shot_personalize','evaluate_personalization',
    'fgsm_attack','pgd_attack','evaluate_adversarial_robustness','plot_adversarial_robustness',
    'detect_transitions','evaluate_transition_accuracy','plot_transition_analysis',
    'simulate_bias_drift','simulate_scale_drift','simulate_noise_drift','evaluate_drift_robustness','plot_drift_robustness',
    'count_macs','estimate_energy','compare_models_energy','plot_energy_comparison',
    'reservoir_manifold_mixup',
    'device','REPO_PATH',
}

# Names defined within notebook cells we'll track
cell_defs = {}

issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    # Find names used
    names = set(re.findall(r'\b([A-Z][A-Za-z_][A-Za-z0-9_]*)\b', src))
    # Find names defined  
    defined_here = set(re.findall(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=', src, re.MULTILINE))
    defined_here |= set(re.findall(r'def\s+([A-Za-z_][A-Za-z0-9_]*)', src))
    defined_here |= set(re.findall(r'class\s+([A-Za-z_][A-Za-z0-9_]*)', src))
    
    # Check uppercase-starting names not yet known
    for name in names:
        if name in known:
            continue
        if name in defined_here:
            continue
        if name in cell_defs:
            continue
        # Skip common builtins
        if name in {'True','False','None','Dataset','TorchDataset','SimpleDataset'}:
            continue
        issues.append((i, name))
    
    for name in defined_here:
        cell_defs[name] = i
    known |= defined_here

print(f"Total cells: {len(nb['cells'])}")
print(f"\nPotentially undefined names by cell:")
seen = {}
for i, name in issues:
    seen.setdefault(i, set()).add(name)
for i in sorted(seen):
    print(f"  Cell {i}: {sorted(seen[i])}")
