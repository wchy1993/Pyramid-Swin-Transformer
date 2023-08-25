# Model configuration
hidden_dim = 96
layers = (8, 6, 4, 4)
heads = (3, 6, 12, 24)
window_sizes = ((4, 8, 16, 32), (4, 8, 16), (4, 8), (4, 8))

# Training configuration
num_epochs = 1
train_batch_size = 8
val_batch_size = 8
learning_rate = 0.001
weight_decay = 0.05
traindir = 'C:\\Users\\wchy1\\Desktop\\Pyramid Swin Transformer\\imagenet\\train'
valdir = 'C:\\Users\\wchy1\\Desktop\\Pyramid Swin Transformer\\imagenet\\val'
