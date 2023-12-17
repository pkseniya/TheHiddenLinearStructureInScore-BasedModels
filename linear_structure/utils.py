import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from pathlib import Path

# from tensordict import TensorDict

def load_dataset(name):
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    if name == "CIFAR10":
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"No {name}")

    return dataset

def isotropic_score(dataset):
    ims = []
    for im, label in dataset:
        ims.append(im)

    X = torch.stack(ims)
    mu = torch.mean(X, dim=0)
    return {"mu": mu}

def truncate_svd(U, lambdas, lambda_min=1e-6):
    mask = lambdas > lambda_min
    lambdas = lambdas[mask]
    U = U[:, mask]
    return U, lambdas

def gaussian_score(dataset):
    ims = []
    for im, label in dataset:
        ims.append(im.flatten())

    X = torch.stack(ims)
    mu = torch.mean(X, dim=0)
    Sigma = torch.cov(X.T)

    U, lambdas, _ = torch.svd(Sigma)
    U, lambdas = truncate_svd(U, lambdas)
    return {"mu": mu, "U": U, "lambdas": lambdas}

def save_parameters(parameters, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, parameter in parameters.items(): 
        torch.save(parameter, save_dir / f"{name}.pt")

def get_gaussian_score(x, sigma, mu, lambdas, U):
    I = torch.eye(U.shape[0], device=U.device)
    lambda_sigma = torch.diag(lambdas / (lambdas + sigma**2))
    score = torch.bmm((I - U @ lambda_sigma @ U.T).unsqueeze(0).expand(x.shape[0], -1, -1), (x - mu).flatten(1).unsqueeze(-1)).squeeze(-1) / sigma**2

    return score

