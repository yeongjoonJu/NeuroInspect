## PATH TO METASHIFT DATASET
import os
METASHIFT_ROOT = "/workspace/neuron_based_debugging/exp_spurious/MetaShift"
CANDIDATES = os.path.join(METASHIFT_ROOT, "dataset/meta_data/full-candidate-subsets.pkl")
METASHIFT_IMAGES = os.path.join(METASHIFT_ROOT, "data/allImages/images/")

CLASSES = ["cat", "dog", "bear", "bird", "elephant"]