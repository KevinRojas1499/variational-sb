from utils.models import MLP, ToyPolicy, LinearMLP


def get_model(name, device):
    # Returns model, ema
    if name == 'mlp':
        return MLP(2,False).requires_grad_(True).to(device=device), \
            MLP(2, False).requires_grad_(False).to(device=device)
    elif name == 'toy':
        return ToyPolicy().requires_grad_(True).to(device=device), \
            ToyPolicy().requires_grad_(False).to(device=device)
    elif name == 'linear':
        return LinearMLP(2,False).requires_grad_(True).to(device=device), \
            LinearMLP(2, False).requires_grad_(False).to(device=device)