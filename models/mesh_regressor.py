import torch
from . import networks
from os.path import join
from util.util import print_network


class RegressionModel:
    """Class for training a regression model on mesh data.

    :args opt: Configuration options including dataset mode, network architecture, etc.
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.loss = None

        # Number of regression outputs (e.g., capacitance values)
        self.noutputs = opt.noutputs

        # Initialize the network
        self.net = networks.define_classifier(
            input_nc=opt.input_nc,
            ncf=opt.ncf,
            ninput_edges=opt.ninput_edges,
            nclasses=self.noutputs,  # For regression, 'nclasses' corresponds to 'noutputs'
            opt=opt,
            gpu_ids=self.gpu_ids,
            arch=opt.arch,
            init_type=opt.init_type,
            init_gain=opt.init_gain
        )
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).float()  # Ensure labels are float tensors
        # Set inputs
        self.edge_features = input_edge_features.to(self.device)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']

        # No need for 'requires_grad' with the latest versions of PyTorch when using optimizers

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    def test(self):
        """Tests the model and returns error metrics."""
        with torch.no_grad():
            out = self.forward()
            error = self.get_error(out, self.labels)
        return error

    def get_error(self, pred, labels):
        """Computes error metric for regression."""
        error = torch.abs(pred - labels).mean()
        return error.item()

    def save_network(self, which_epoch):
        """Save the model to disk."""
        save_filename = f'{which_epoch}_net.pth'
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def load_network(self, which_epoch):
        """Load the model from disk."""
        save_filename = f'{which_epoch}_net.pth'
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print(f'Loading the model from {load_path}')
        state_dict = torch.load(load_path, map_location=self.device)
        net.load_state_dict(state_dict)

    def update_learning_rate(self):
        """Update learning rate (called once every epoch)."""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print(f'Learning rate = {lr:.7f}') 