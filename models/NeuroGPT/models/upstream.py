"""
NeuroGPT Original Model Components

This module imports and preserves the original NeuroGPT core components:
- Encoder: based on braindecode EEG encoder
- CSMEmbedder: embedder that projects to Transformer space
- GPTModel: GPT2-based decoder
- Model: Main model combining encoder, embedder, and decoder
"""
import os
import sys
import importlib.util

# Get the NeuroGPT source directory
# models/NeuroGPT/models/upstream.py -> models/NeuroGPT/models/src
_current_file = os.path.abspath(__file__)
_models_dir = os.path.dirname(_current_file)  # models/NeuroGPT/models
NEUROGPT_SRC_DIR = os.path.normpath(os.path.join(_models_dir, 'src'))

# We need to add the src directory to sys.path so that the encoder, embedder,
# decoder packages can find each other
if NEUROGPT_SRC_DIR not in sys.path:
    sys.path.insert(0, NEUROGPT_SRC_DIR)

# Direct import using absolute import with spec (the path is needed for internal imports)
def import_module_from_path(module_name, file_path):
    """Import a module directly from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import encoder
encoder_path = os.path.join(NEUROGPT_SRC_DIR, 'encoder', 'conformer_braindecode.py')
encoder_mod = import_module_from_path('encoder_conformer', encoder_path)
Encoder = encoder_mod.EEGConformer

# Import embedder
embedder_path = os.path.join(NEUROGPT_SRC_DIR, 'embedder', 'csm.py')
embedder_mod = import_module_from_path('embedder_csm', embedder_path)
CSMEmbedder = embedder_mod.CSMEmbedder

# Import decoder
decoder_path = os.path.join(NEUROGPT_SRC_DIR, 'decoder', 'gpt.py')
decoder_mod = import_module_from_path('decoder_gpt', decoder_path)
GPTModel = decoder_mod.GPTModel

# Import model
model_path = os.path.join(NEUROGPT_SRC_DIR, 'model.py')
model_mod = import_module_from_path('neurogpt_model', model_path)
NeuroGPTModel = model_mod.Model

# After all imports are done, remove the path from sys.path to avoid conflicts
# (but keep modules in sys.modules for later use)
if NEUROGPT_SRC_DIR in sys.path:
    sys.path.remove(NEUROGPT_SRC_DIR)

__all__ = ['Encoder', 'CSMEmbedder', 'GPTModel', 'NeuroGPTModel']
