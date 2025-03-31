import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################

class ResNet18TransferNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18TransferNet, self).__init__()
        # Resize layer to upscale 32x32 images to 224x224
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        # Load pretrained ResNet18
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # Replace the final fully connected layer to match our number of classes
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.resize(x)  # Resize to (batch, 3, 224, 224)
        x = self.resnet(x)  # Forward pass through ResNet18
        return x
    
    def freeze_base_network(self):
        """Freeze all layers except the final FC layer"""
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze the FC layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_layers(self, from_layer=None):
        """Unfreeze layers from the specified layer onwards"""
        if from_layer is None:
            # Unfreeze all layers
            for param in self.resnet.parameters():
                param.requires_grad = True
        else:
            # Keep initial layers frozen and unfreeze later layers
            layers_to_unfreeze = False
            for name, child in self.resnet.named_children():
                if name == from_layer:
                    layers_to_unfreeze = True
                    
                if layers_to_unfreeze:
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False

class ResNet34TransferNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet34TransferNet, self).__init__()
        # Resize layer to upscale 32x32 images to 224x224
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        # Load pretrained ResNet34
        self.resnet = torchvision.models.resnet34(pretrained=True)
        # Replace the final fully connected layer to match our number of classes
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.resize(x)   # Resize to (batch, 3, 224, 224)
        x = self.resnet(x)   # Forward pass through ResNet34
        return x
    
    def freeze_base_network(self):
        """Freeze all layers except the final FC layer"""
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze the FC layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_layers(self, from_layer=None):
        """Unfreeze layers from the specified layer onwards"""
        if from_layer is None:
            # Unfreeze all layers
            for param in self.resnet.parameters():
                param.requires_grad = True
        else:
            # Keep initial layers frozen and unfreeze later layers
            layers_to_unfreeze = False
            for name, child in self.resnet.named_children():
                if name == from_layer:
                    layers_to_unfreeze = True
                    
                if layers_to_unfreeze:
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False



################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()   ### TODO
        _, predicted = torch.max(outputs.data, 1)     ### TODO

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    print(train_acc)
    return train_loss, train_acc

################################################################################
# Early stopping to stop training early if loss plateaus
################################################################################

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    Attributes:
        patience (int): How many epochs to wait after last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path for the checkpoint to be saved to.
        counter (int): Number of epochs since the last improvement.
        best_score (float): Best score achieved so far.
        early_stop (bool): Whether to stop the training.
        val_loss_min (float): Minimum validation loss encountered so far.
    """
    
    def __init__(self, patience=5, verbose=True, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        # Invert the loss because lower loss is better.
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta: # Count increases if the epoch does not produce significant improvement in validation loss
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: # If the patience is exceeded, training stops
                if self.verbose:
                    print("Early stopping triggered")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model) # Model is saved in checkpoints to preserve the best model in the case of early stopping
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) ### TODO -- inference
            loss = criterion(outputs, labels)    ### TODO -- loss calculation

            running_loss += loss.item()  ### SOLUTION -- add loss from this sample
            _, predicted = torch.max(outputs.data, 1)   ### SOLUTION -- predict the class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def apply_fine_tuning_strategy(model, epoch, optimizer, CONFIG):
    """
    Apply the appropriate fine-tuning strategy based on the current epoch
    and configuration. Returns a new optimizer if needed.
    """
    if not CONFIG["fine_tuning"]["enabled"]:
        return optimizer
        
    ft_config = CONFIG["fine_tuning"]
    strategy = ft_config["strategy"]
    architecture = CONFIG["architecture"]
    
    # Two-stage strategy: unfreeze deeper layers after specified epochs
    if strategy == "two_stage" and epoch == ft_config["unfreeze_after"]:
        print('Unfreezing all layers...')
        model.unfreeze_layers()
        
        # Create new optimizer with different learning rates
        if architecture in ["resnet18", "resnet34"]:
            optimizer = optim.AdamW([
                {'params': [p for n, p in model.resnet.named_parameters() 
                           if 'fc' not in n], 'lr': ft_config["base_lr"]},
                {'params': model.resnet.fc.parameters(), 'lr': ft_config["new_layers_lr"]}
            ], weight_decay=CONFIG["weight_decay"])
        
        return optimizer
    
    # Gradual strategy: progressively unfreeze deeper layers
    elif strategy == "gradual":
        epochs = CONFIG["epochs"]
        
        if epoch == int(epochs * 0.2):
            if architecture in ["resnet18", "resnet34"]:
                print('Unfreezing layer4...')
                model.unfreeze_layers(from_layer='layer4')
                optimizer = optim.AdamW([
                    {'params': model.resnet.layer4.parameters(), 'lr': ft_config["base_lr"]},
                    {'params': model.resnet.fc.parameters(), 'lr': ft_config["new_layers_lr"]}
                ], weight_decay=CONFIG["weight_decay"])
            return optimizer
            
        elif epoch == int(epochs * 0.4):
            if architecture in ["resnet18", "resnet34"]:
                print('Unfreezing layer3...')
                model.unfreeze_layers(from_layer='layer3')
                optimizer = optim.AdamW([
                    {'params': model.resnet.layer3.parameters(), 'lr': ft_config["base_lr"] * 0.5},
                    {'params': model.resnet.layer4.parameters(), 'lr': ft_config["base_lr"]},
                    {'params': model.resnet.fc.parameters(), 'lr': ft_config["new_layers_lr"]}
                ], weight_decay=CONFIG["weight_decay"])

            return optimizer
    
    # Return the original optimizer if no changes
    return optimizer

def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "architecture": "resnet34",   # Options: "resnet18", "resnet34"
        "batch_size": 128, # run batch size finder to find optimal batch size
        "learning_rate": 0.01,
        "epochs": 50,  # Train for longer in a real scenario
        "num_workers": 1, # Adjust based on your system
        "device": "cuda",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "weight_decay": 0.001,
        "label_smoothing": 0.1,
        
        # Fine-tuning configuration
        "fine_tuning": {
            "enabled": True,
            "strategy": "full",  # Options: "two_stage", "gradual", "full"
            "unfreeze_after": 5,      # For two_stage: after how many epochs to unfreeze layers
            "base_lr": 0.0001,        # Learning rate for pre-trained layers
            "new_layers_lr": 0.001    # Learning rate for new/unfrozen layers
        }
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with 50% probability
        # transforms.RandomVerticalFlip(p=0.5),    # Randomly flip images vertically with 50% probability
        transforms.RandomRotation(degrees=15),   # Randomly rotate images within a range of ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),   # Apply affine transformations
        transforms.ToTensor(),                   # Convert PIL image to tensor
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize the image
    ])


    ###############
    # TODO Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # Example normalization
    ])

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(len(trainset) * .8)   ### Rounded with int function
    val_size = len(trainset) - train_size     ### TODO -- Calculate validation set size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])  ### TODO -- split into training and validation sets

    ### TODO -- define loaders and test set
    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=CONFIG['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG['batch_size'], shuffle=False)

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    if CONFIG["architecture"] == "resnet18":
        model = ResNet18TransferNet(num_classes=100)
    elif CONFIG["architecture"] == "resnet34":
        model = ResNet34TransferNet(num_classes=100)
    else:
        raise ValueError(f"Unsupported architecture: {CONFIG['architecture']}")
        
    model = model.to(CONFIG["device"])   # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    
    ############################################################################
    # Initialize fine-tuning (if enabled)
    ############################################################################
    if CONFIG["fine_tuning"]["enabled"]:
        strategy = CONFIG["fine_tuning"]["strategy"]
        print(f"Initializing fine-tuning with '{strategy}' strategy")
        
        if strategy == "two_stage" or strategy == "gradual":
            # Start with frozen base network
            model.freeze_base_network()
            # Initial optimizer targets only unfrozen parts (FC layer or classifier)
            if CONFIG["architecture"] in ["resnet18", "resnet34"]:
                optimizer = optim.AdamW([
                    {'params': model.resnet.fc.parameters(), 
                     'lr': CONFIG["fine_tuning"]["new_layers_lr"]}
                ], weight_decay=CONFIG["weight_decay"])
        else:  # "full" strategy
            # Fine-tune the whole network with different learning rates
            if CONFIG["architecture"] in ["resnet18", "resnet34"]:
                optimizer = optim.AdamW([
                    {'params': [p for n, p in model.resnet.named_parameters() 
                               if 'fc' not in n], 
                     'lr': CONFIG["fine_tuning"]["base_lr"]},
                    {'params': model.resnet.fc.parameters(), 
                     'lr': CONFIG["fine_tuning"]["new_layers_lr"]}
                ], weight_decay=CONFIG["weight_decay"])
    else:
        # Standard optimizer initialization
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])   # Experiment with label smoothing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])  # Add a scheduler

    # Initialize wandb
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    early_stopping = EarlyStopping(patience=5, verbose=True, path='best_model.pt') # Initialize early stopping

    for epoch in range(CONFIG["epochs"]):
        # Apply fine-tuning strategy (only applies changes when needed)
        if CONFIG["fine_tuning"]["enabled"]:
            optimizer = apply_fine_tuning_strategy(model, epoch, optimizer, CONFIG)
            # Create a new scheduler for the updated optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CONFIG['epochs']-epoch)
                
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Save to wandb as well

        early_stopping(val_loss, model) # Determine if early stopping should be done
        if early_stopping.early_stop:
            print("Stopping early at epoch", epoch+1)
            break 

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()
