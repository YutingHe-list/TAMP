import sys
import os
import numpy as np
import torch
import argparse
from adan import Adan
import torch.optim.lr_scheduler as lr_scheduler

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network_MITNet import MITNet as my_network
from utils.MyLoss import MyLoss as my_loss
from utils.MyDataset import MyDataset as my_dataset


def get_parser():
    parser = argparse.ArgumentParser(description='TAMP Pre-training on SimNICT datasets')

    # Data paths
    parser.add_argument('--input_folder', type=str, default="samples/pretrain/input",
                        help="folder containing input NICT volumes")
    parser.add_argument('--label_folder', type=str, default="samples/pretrain/label",
                        help="folder containing label volumes")

    # Training parameters
    parser.add_argument('--training_volumes', type=int, default=100,
                        help="total number of training volumes")
    parser.add_argument('--queue_len', type=int, default=5,
                        help="number of volumes to load in memory queue (N=5)")
    parser.add_argument('--queue_iterate_times', type=int, default=10,
                        help="number of times to iterate through all volumes")
    parser.add_argument('--nii_start_index', type=int, default=1,
                        help="starting index for loading volumes")
    
    # Batch and device
    parser.add_argument('--batch_size', type=int, default=8,
                        help="batch size for training")
    parser.add_argument('--cuda_index', type=int, default=0,
                        help="GPU device index")

    # Optimizer parameters (Adan optimizer)
    parser.add_argument('--lr', type=float, default=0.001,
                        help="initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay for Adan optimizer')
    parser.add_argument('--opt_betas', default=[0.98, 0.92, 0.99], type=float, nargs='+',
                        metavar='BETA', help='optimizer betas in Adan')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='optimizer epsilon to avoid division by zero')
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='gradient clipping threshold (0.0 = no clipping)')
    parser.add_argument('--no_prox', action='store_true', default=False,
                        help='whether perform weight decay like AdamW')

    # Learning rate scheduler
    parser.add_argument('--lr_step_size', type=int, default=50,
                        help="learning rate decay step size")
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help="learning rate decay factor")

    # Warm-up parameters
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help="number of warm-up epochs")
    parser.add_argument('--warmup_lr_start', type=float, default=0.0001,
                        help="starting learning rate for warm-up")

    # Checkpoint saving
    parser.add_argument('--save_interval', type=int, default=10,
                        help="save checkpoint every N epochs")
    parser.add_argument('--checkpoint_dir', type=str, default="weights/TAMP_pretrain_checkpoints",
                        help="directory to save checkpoints")

    return parser


def show_training_global_info(nii_epoch, train_loss, total_epochs):
    """Display epoch-level training information"""
    sys.stdout.write(
        f"\n[Epoch {nii_epoch}/{total_epochs}] [loss {train_loss.mloss():.6f}]\n"
    )


def show_training_local_info(nii_epoch, train_loss, batch_idx, total_batches, total_epochs):
    """Display batch-level training information"""
    sys.stdout.write(
        f"\r{' ' * 100}" 
        f"\r[epoch:{nii_epoch}/{total_epochs}] [batch:{batch_idx}/{total_batches}] "
        f"[loss:{train_loss.mloss():.6f} = MSE:{train_loss.mmse():.6f} + "
        f"VGG:{train_loss.mvgg():.6f} + SSIM:{train_loss.mssim():.6f} + PROJ:{train_loss.mprom():.6f}]"
    )
    sys.stdout.flush()


def unstandard(standard_img):
    """Convert standardized image back to original scale"""
    mean = -556.882367
    variance = 225653.408219
    nii_slice = standard_img * np.sqrt(variance) + mean
    return nii_slice


def standard(nii_slice):
    """Standardize image to zero mean and unit variance"""
    mean = -556.882367
    variance = 225653.408219
    nii_slice = nii_slice.astype(np.float32)
    nii_slice = (nii_slice - mean) / np.sqrt(variance)
    return nii_slice


def load_model(opt):
    """Initialize the MITNet model"""
    model = my_network(img_size=512)
    model.to(f"cuda:{opt.cuda_index}")
    
    # Print model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Model: MITNet")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}\n")
    
    return model


def get_warmup_lr(epoch, opt):
    """Calculate learning rate during warm-up phase"""
    if epoch >= opt.warmup_epochs:
        return opt.lr
    else:
        # Linear warm-up from warmup_lr_start to lr
        return opt.warmup_lr_start + (opt.lr - opt.warmup_lr_start) * (epoch / opt.warmup_epochs)


def adjust_learning_rate(optimizer, epoch, opt):
    """Adjust learning rate with warm-up"""
    lr = get_warmup_lr(epoch, opt)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(model, optimizer, scheduler, epoch, opt, filename):
    """Save model checkpoint"""
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(opt.checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")


def pretrain(opt):
    """Main pre-training function with queued training process"""
    
    print("\n" + "="*60)
    print("TAMP Pre-training on SimNICT Datasets")
    print("="*60)
    print(f"Training volumes: {opt.training_volumes}")
    print(f"Queue length: {opt.queue_len}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Initial learning rate: {opt.lr}")
    print(f"Warm-up epochs: {opt.warmup_epochs}")
    print(f"Device: cuda:{opt.cuda_index}")
    print("="*60 + "\n")
    
    # Initialize model, dataset, and loss
    model = load_model(opt)
    train_dataset = my_dataset(opt)
    train_loss = my_loss()
    
    # Initialize optimizer (Adan)
    optimizer = Adan(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        betas=opt.opt_betas,
        eps=opt.opt_eps,
        max_grad_norm=opt.max_grad_norm,
        no_prox=opt.no_prox
    )
    
    # Initialize learning rate scheduler (applied after warm-up)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=opt.lr_step_size,
        gamma=opt.lr_gamma
    )
    
    # Calculate total epochs
    total_epochs = opt.training_volumes * opt.queue_iterate_times
    
    print("Starting training...\n")
    
    # Training loop with queued data loading
    for nii_epoch in range(opt.nii_start_index, total_epochs + 1):
        
        # Adjust learning rate with warm-up
        if nii_epoch <= opt.warmup_epochs:
            current_lr = adjust_learning_rate(optimizer, nii_epoch, opt)
            print(f"Warm-up phase: epoch {nii_epoch}/{opt.warmup_epochs}, lr={current_lr:.6f}")
        
        # Create data loader for current queue
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=False  # Already shuffled in dataset
        )
        total_batches = len(train_loader)
        
        # Set model to training mode
        model.train()
        
        # Iterate through batches
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            
            # Add channel dimension and move to GPU
            inputs = inputs.unsqueeze(1).to(f"cuda:{opt.cuda_index}")
            labels = labels.unsqueeze(1).to(f"cuda:{opt.cuda_index}")
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = train_loss.cal_loss(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Display batch-level information
            show_training_local_info(nii_epoch, train_loss, batch_idx + 1, total_batches, total_epochs)
        
        # Display epoch-level information
        show_training_global_info(nii_epoch, train_loss, total_epochs)
        
        # Save checkpoint at intervals
        if nii_epoch % opt.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, nii_epoch, opt,
                f"checkpoint_epoch_{nii_epoch}.pkl"
            )
        
        # Learning rate scheduling (after warm-up)
        if nii_epoch > opt.warmup_epochs:
            scheduler.step()
        
        # Refresh queue: remove oldest volume and load new one
        train_dataset.refresh_next_train()
        train_loss.clear()
    
    # Save final model
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60 + "\n")
    
    final_model_path = "weights/TAMP_pretrain_weight/TAMP_pretrain.pkl"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}\n")


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    
    # Start pre-training
    pretrain(opt)
