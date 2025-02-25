import argparse

# Import custom modules
from utils import *
from preprocessing import *
from ad_model import *
from ad_trainer import *
from cfdet import *
from cfdet_trainer import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Anomaly Detection and Explanation')
parser.add_argument('--dataset_path', type=str, default='./data/BGL.log_structured_v1.csv', help='Path to the dataset')
parser.add_argument('--model_path', type=str, default='./output/model/', help='Path to save/load models')
parser.add_argument('--output_path', type=str, default='./output/', help='Path to save evaluation results')

# Batch sizes for different phases of training and evaluation
parser.add_argument('--batch_train', type=int, default=512, help='Batch size for training')
parser.add_argument('--batch_test', type=int, default=4096, help='Batch size for testing')
parser.add_argument('--batch_val', type=int, default=4096, help='Batch size for validation')
parser.add_argument('--batch_train_test', type=int, default=1024, help='Batch size for train-test split')

# Hyperparameters for Deep SVDD anomaly detection model
parser.add_argument('--embedding_dim', type=float, default=50, help='Embedding dimension for Deep SVDD')
parser.add_argument('--hidden_dim', type=float, default=128, help='Hidden dimension for Deep SVDD')
parser.add_argument('--num_layers', type=float, default=1, help='Number of layers for Deep SVDD')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for anomaly detection model training')

# Hyperparameters for CFDet explanation model
parser.add_argument('--embedding_dim2', type=float, default=100, help='Embedding dimension for CFDet')
parser.add_argument('--hidden_dim2', type=float, default=128, help='Hidden dimension for CFDet')
parser.add_argument('--num_layers2', type=float, default=1, help='Number of layers for CFDet')
parser.add_argument('--epochs2', type=int, default=100, help='Number of epochs for explanation model training')

# Regularization parameters for explanation model
parser.add_argument('--triplet_lambda', type=float, default=1.0, help='Triplet loss weight')
parser.add_argument('--sparsity_lambda', type=float, default=0.15, help='Sparsity loss weight')
parser.add_argument('--continuity_lambda', type=float, default=0.05, help='Continuity loss weight')

# Learning rate for optimization
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--state_dict_path', type=str, default='state_dict_minloss.bin',
                    help='Path to save/load trained generator')
args = parser.parse_args()

# Set computing device (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
setup_seed(10)  # Set seed for reproducibility

# Data Processing
print("Starting data processing...")
data_processor = DataProcessor(args.dataset_path, args.batch_train, args.batch_test, args.batch_val,
                               args.batch_train_test)
train_loader, test_loader, val_loader = data_processor.train_loader, data_processor.test_loader, data_processor.val_loader
print("Data processing completed successfully.\n")

# Initialize Anomaly Detection Model
print("Initializing anomaly detection model (SVDD)...")
ad_model = SVDD(data_processor.vocab_size, device, args.embedding_dim, args.hidden_dim, args.num_layers).to(device)
ad_trainer = SVDDTrainer(ad_model, train_loader, device, args.batch_train, args.epochs)

# Train and evaluate the anomaly detection model
print("Starting anomaly detection model training...")
ad_trainer.train(args.model_path)
print("Anomaly detection training completed.")

print("Evaluating anomaly detection model...")
ad_trainer.evaluate(val_loader, args.output_path + 'ad_result/val.txt')
print("Anomaly detection model evaluation completed.\n")

# Find baseline sequence based on Deep SVDD model characteristics
print("Extracting baseline sequence...")
baseline_sequence = find_baseline_sequence(ad_model, ad_trainer.center, ad_trainer.radius, train_loader, device)
print("Baseline sequence extraction completed.\n")

# Split test sequences into different distance-based categories
print("Splitting sequences based on distance...")
sequences, labels, keys = split_sequences_by_distance(ad_model, ad_trainer.center, ad_trainer.radius, test_loader,
                                                      device)
print("Sequence splitting completed.\n")

# Prepare data loaders for CFDet training
print("Preparing data loaders for counterfactual explanation training...")
train_test_loader_far, train_test_data_far = train_test_data_loader(sequences['far'], labels['far'], keys['far'],
                                                                    args.batch_train_test)
train_test_loader_medium, train_test_data_medium = train_test_data_loader(sequences['medium'], labels['medium'],
                                                                          keys['medium'], args.batch_train_test)
train_test_loader_close, train_test_data_close = train_test_data_loader(sequences['close'], labels['close'],
                                                                        keys['close'], args.batch_train_test)
print("Data loaders prepared.\n")

# Initialize Explanation Model (CFDet)
print("Initializing CFDet counterfactual explanation model...")
generator = Generator(data_processor.vocab_size, device, args.embedding_dim2, args.hidden_dim2, args.num_layers2)
cfdet = CFDet(generator, device).to(device)
print("CFDet model initialized.\n")

# Train CFDet model for counterfactual explanation
print("Starting CFDet model training for counterfactual explanations...")
cfdet_trainer = CFDetTrainer(cfdet, ad_model, train_test_loader_far, args.batch_train_test, args.epochs2,
                             ad_trainer.center, device, baseline_sequence, args.triplet_lambda, args.sparsity_lambda,
                             args.continuity_lambda)
cfdet_trainer.train(args.model_path)
print("CFDet model training completed.\n")

# Evaluate CFDet model on validation and test datasets
print("Evaluating CFDet model on validation dataset...")
cfdet_trainer.evaluate(val_loader, args.output_path + 'explain_result/eval.txt')
print("Validation evaluation completed.\n")

print("Evaluating CFDet model on test dataset...")
cfdet_trainer.evaluate(test_loader, args.output_path + 'explain_result/test.txt')
print("Test evaluation completed.\n")

print("Training and evaluation completed successfully.")
