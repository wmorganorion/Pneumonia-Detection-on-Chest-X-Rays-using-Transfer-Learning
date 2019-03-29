import time
import os
import copy

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from torchvision import datasets, transforms, models



def load_model(arch='vgg13', num_labels=10, hidden_units=2048):
    
    # Load a pre-trained model
    if arch=='vgg13':
        # Load a pre-trained model
        model=models.vgg13(pretrained=True)    
    elif arch=='alexnet':
        model=models.alexnet(pretrained=True)      
    else:
        raise ValueError('Unexpected pre-trained network error', arch)
        
    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Features, removing the last layer
    features = list(model.classifier.children())[:-1]
  
    # Number of filters in the bottleneck layer
    num_filters = model.classifier[len(features)].in_features

    # Extend the existing architecture with new layers
    features.extend([
        nn.Dropout(),
        nn.Linear(num_filters, hidden_units),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, num_labels),
        
    ])
    
    model.classifier = nn.Sequential(*features)

    print('Model loaded : ', arch)

    return model


def train_model(image_datasets, arch='vgg13', hidden_units=2048, checkpoint='', epochs=1, learning_rate=0.001, gpu=False):
            
    print(arch, epochs, learning_rate, gpu)

    # Using the image datasets, define the dataloaders
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2) 
                   for x in list(image_datasets.keys())}
 
    # Calculate dataset sizes.
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in list(image_datasets.keys())}    

        
    print('pre-trained network [arch]:', arch)
    print('hidden units:', hidden_units)
    print('Learning rate:', learning_rate)
    print('epochs:', epochs)
    

    # Load the model     
    num_labels = len(image_datasets['train'].classes)
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")     

                
    # Defining criterion, optimizer and scheduler    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)    
        
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    # Save checkpoint if requested
    if checkpoint:
        print ('Saving checkpoint to:', checkpoint) 
        checkpoint_dict = {
            'arch': arch,
            'class_to_idx': model.class_to_idx, 
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units
        }
        
        torch.save(checkpoint_dict, checkpoint)
        
    # Return the model
    return model

