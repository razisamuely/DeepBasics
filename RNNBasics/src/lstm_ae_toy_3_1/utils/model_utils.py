import torch
import torch.nn as nn
from torch import optim
import json

def get_optimizer(optimizer_name: str, parameters, lr: float):
    """Get optimizer by name."""
    optimizer_dict = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'adamw': optim.AdamW
    }
    return optimizer_dict[optimizer_name.lower()](parameters, lr=lr)

def get_loss_function(loss_name: str):
    """Get loss function by name."""
    loss_dict = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smoothl1': nn.SmoothL1Loss()
    }
    return loss_dict[loss_name.lower()]

def save_model_results(study, model, test_loss, args, results_dir):
    """Save model results and state dict."""
    test_loss_value = test_loss.item() if torch.is_tensor(test_loss) else float(test_loss)
    
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'test_loss': test_loss_value,
        'best_trial_number': study.best_trial.number,
        'configuration': {
            'optimizer': args.optimizer,
            'loss': args.loss,
            'search_space': {
                'hidden_size_range': [args.hidden_size_min, args.hidden_size_max],
                'learning_rate_range': [args.lr_min, args.lr_max],
                'grad_clip_range': [args.grad_clip_min, args.grad_clip_max]
            }
        }
    }
    
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    torch.save(model.state_dict(), results_dir / 'best_model.pth')