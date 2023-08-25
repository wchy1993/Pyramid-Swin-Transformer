# Model configuration
hidden_dim = 96
layers = (8, 6, 4, 4)
heads = (3, 6, 12, 24)
window_sizes = ((4, 8, 16, 32), (4, 8, 16), (4, 8), (4, 8))

# Training configuration
num_epochs = 10
train_batch_size = 32
val_batch_size = 32
learning_rate = 0.001
weight_decay = 0.05
traindir = '/imagenet/train'
valdir = '/imagenet/val'
