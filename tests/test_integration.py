#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:04:42 2026

@author: jbrennan
"""

import os
import shutil
import tempfile
import unittest

from pathlib import Path

from deeponet_chem.train_setup import start_training_run
from deeponet_chem.mock_data import generate_mock_dataset


class TestTrainingIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Use `SHOW_TEST_OUTPUT=1 python -m unittest` to show test std output.
        cls.show_output = os.getenv('SHOW_TEST_OUTPUT', '0') == '1'
    
    
    @classmethod
    def tearDownClass(cls):
        pycache = Path('__pycache__')
        if pycache.exists():
            shutil.rmtree(pycache)
    
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir='.')
        self.test_output_dir = Path(self.temp_dir.name)
        
        # Configure mock dimensions based on your TOML specs
        self.num_tracers = 35 
        self.num_hydro_steps = 35
        self.chem_per_hydro = 5
        self.n_params = 7 
        self.n_species = 14
        
        self.paths = {
            "data_dir": str(self.test_output_dir / 'data'),
            "metadata_file": "metadata.json",
            "train_data": "train_dataset.npy",
            "valid_data": "valid_dataset.npy",
            "checkpoints": str(self.test_output_dir / 'checkpoints')
        }
        
        self.model_cfg = {
            "input_size":    self.n_params + self.n_species,
            "output_size":   self.n_species,
            "latent_size":   16,
            "branch_size":   32,
            "trunk_size":    32,
            "branch_layers": 2,
            "trunk_layers":  2,
            "activation":    "tanh",
        }
        
        self.train_cfg = {
            "batch_size":    7,
            "learning_rate": 1e-3,
            "weight_decay":  1e-4,
            "num_epochs":    2,
            "shuffle_train": True,
            "validators":    ['OneStep', 'Rollout'],
            "primary_loss":  'one_step_loss'
        }
    
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    
    def test_training_smoke(self):
        """Smoke test to run a couple of epochs with mock data."""
        
        # Generate mock train and validation datasets
        generate_mock_dataset(
            data_path       = self.test_output_dir / 'data',
            num_tracers     = self.num_tracers,
            num_hydro_steps = self.num_hydro_steps,
            chem_per_hydro  = self.chem_per_hydro,
            n_params        = self.n_params,
            n_species       = self.n_species
        )
        
        # Run brief training loop
        start_training_run(self.paths, self.model_cfg, self.train_cfg)
        
        # Assertions to verify correct execution and checkpoint creation
        checkpoint_path = self.test_output_dir / "checkpoints"
        self.assertTrue(
            checkpoint_path.exists(),
            "Checkpoints root directory was not created."
        )
        
        checkpoint_dirs = list(checkpoint_path.glob('best_model_*'))
        self.assertTrue(
            len(checkpoint_dirs) > 0,
            "No checkpoint subdirectories found."
        )
        
        # Test restarting from a checkpoint.
        restart_checkpoint_path = self.test_output_dir / "restart_checkpoint"
        checkpoint_path = checkpoint_dirs[0]
        train_cfg = self.train_cfg | {
            'model_checkpoint'     : checkpoint_path / 'model.pt',
            'optimizer_checkpoint' : checkpoint_path / 'optimizer.pt'
        }
        paths = self.paths | {'checkpoints' : restart_checkpoint_path}
        start_training_run(paths, self.model_cfg, train_cfg)
        
        # Assert correct execution and checkpoint creation
        self.assertTrue(
            restart_checkpoint_path.exists(),
            "Restart checkpoint path was not created."
        )
        


if __name__ == "__main__":
    unittest.main()