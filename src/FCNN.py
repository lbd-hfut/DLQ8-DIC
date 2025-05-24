import torch
import torch.nn as nn

class RBFCNN(nn.Module):
    def __init__(self, centers, init_b=10.0, device='cpu'):
        super(RBFCNN, self).__init__()
        self.device = torch.device(device)
        self.input_dim = centers.shape[1]
        self.N = centers.shape[0]

        # BatchNorm without affine parameters
        self.norm_x = nn.BatchNorm1d(self.input_dim, affine=False).to(self.device)
        self.norm_c = nn.BatchNorm1d(self.input_dim, affine=False).to(self.device)

        # register raw centers and normalize
        raw_centers = centers.clone().detach().to(self.device)
        self.register_buffer('raw_centers', raw_centers)
        with torch.no_grad():
            self.norm_c.train()  # force BatchNorm to compute stats
            normed_centers = self.norm_c(self.raw_centers)
        self.centers = nn.Parameter(normed_centers, requires_grad=False)

        # Trainable parameters (all moved to correct device)
        self.a_u = nn.Parameter(torch.randn(self.N, 1, device=self.device))
        self.a_v = nn.Parameter(torch.empty(self.N, 1, device=self.device).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.full((self.N, 1), init_b, device=self.device))


    def forward(self, xy):
        """
        :param xy: (B, 2) tensor of input coordinates (x, y)
        :return: (B, 2) tensor of outputs (u, v)
        """
        xy_norm = self.norm_x(xy)  # (B, 2)
        x = xy_norm.unsqueeze(1)        # (B, 1, 2)
        c = self.centers.unsqueeze(0)   # (1, N, 2)
        r2 = ((x - c) ** 2).sum(dim=2)  # (B, N)
        phi = torch.exp(- (self.b.T ** 2) * r2)  # (B, N)

        u = phi @ self.a_u
        v = phi @ self.a_v
        return torch.cat([u, v], dim=1)

    # 以下保留前面你提到的辅助功能
    def Earlystop_set(self, patience=10, delta=0, path=None):
        self.patience = int(patience)
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def Earlystop(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss < val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self):
        if self.path:
            checkpoint = {'model_state_dict': self.state_dict()}
            torch.save(checkpoint, self.path)
            self.best_model = self.state_dict()

    def freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def initialize_weights(self, method='xavier'):
        if method not in ['xavier', 'kaiming']:
            raise ValueError("Unsupported method. Use 'xavier' or 'kaiming'.")
        initializer = nn.init.xavier_uniform_ if method == 'xavier' else nn.init.kaiming_uniform_
        initializer(self.a_u)
        initializer(self.a_v)
        nn.init.constant_(self.b, 10.0)
