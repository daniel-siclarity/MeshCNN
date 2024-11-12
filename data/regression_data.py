import os
import torch
import numpy as np
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh

class RegressionData(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(self.root, opt.phase)
        self.paths = self.make_dataset(self.dir)
        self.size = len(self.paths)
        self.get_mean_std()
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        label = self.load_label(path)
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        edge_features = (edge_features - self.mean) / self.std
        return {
            'mesh': mesh,
            'edge_features': edge_features,
            'label': label
        }

    def __len__(self):
        return self.size

    def make_dataset(self, dir):
        meshes = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)
        return meshes

    def load_label(self, mesh_path):
        # Implement this method to load capacitance values corresponding to the mesh
        # Example: Assume labels are stored in a separate text file with the same name but '_label.txt' extension
        label_path = mesh_path.replace('.obj', '_label.txt')
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file not found for {mesh_path}")
        label = torch.from_numpy(np.loadtxt(label_path)).float()
        return label.to(self.device) 