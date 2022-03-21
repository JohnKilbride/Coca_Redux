import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.segmentation_model import SegmentationModel
import utils.dataset as dataset_utils
from utils.augmentations import load_augmentations
    
def str_to_bool(in_str):
    if str(in_str) == "None":
        return None
    elif str(in_str) == "True":
        return True
    elif str(in_str) == "False":
        return False
    else:
        raise ValueError("str_to_bool -- value for bool should be a string with value 'True' or 'False' (check ya' argparse).")

def cli_main (args):
    
    # ------------
    # data
    # ------------
    # Define the transforms
    args.augs = load_augmentations(args.use_noise)
    
    # Load in the dataset
    folds = dataset_utils.partitions_to_folds(args.seed, args.train_data_dir, args.num_folds)

    # Loop over the different data partitions
    for fold_i, fold in enumerate(folds):
        
        print("\n============\n   Fold {i} of {tot}  \n=========\n".format(i=fold_i+1, tot=args.num_folds))
        args.test_partitions = fold
        
        # Define the transforms
        args.augs = load_augmentations(args.use_noise)
        
        # Load in the dataset
        train_dataset, test_dataset = dataset_utils.load_fold_train_test(
            fold, args.train_data_dir, args.test_data_dir, args.norm_stats_csv, args.deformed_data_dir, args.augs
            )
    
        # Define the DataLoader
        train_loader = DataLoader(train_dataset, 
                                  batch_size = args.batch_size, 
                                  shuffle = True,
                                  num_workers = args.num_workers,
                                  persistent_workers = False,
                                  pin_memory = True,
                                  )    
        test_loader = DataLoader(test_dataset, 
                                  batch_size = args.batch_size, 
                                  shuffle = False,
                                  num_workers = 0,
                                  persistent_workers = False,
                                  pin_memory = True,
                                  )      
    
        # ------------
        # logging
        # ------------  
        # Add the weights and biases logger
        model_name = '{encoder}-{decoder}-{loss_name}-{lr}-{batch}-{resolution}-{noise}-{suffix}-fold={i}'
        args.model_name = model_name.format(
            encoder = args.encoder,
            decoder = args.decoder,
            loss_name = args.loss_name,
            lr = args.learning_rate,
            batch = args.batch_size,
            resolution = args.resolution,
            noise = str(args.use_noise),
            suffix = args.suffix,
            i = fold_i 
            )
        logger = TensorBoardLogger(
            name = args.model_name,
            save_dir = args.log_dir,
            )       

        # ------------
        # Load Model
        # ------------
        # Seed everything
        pl.seed_everything(args.seed, workers=True)
        
        # Construct the model
        dict_args = vars(args)
        net = SegmentationModel(**dict_args)
        
        # First callback: early stopping
        callback1 = EarlyStopping(
            monitor = "val_logger_loss", 
            min_delta=0.001, 
            patience=10, 
            verbose=False, 
            mode="min"
            )
        
        # Second callback: This will log, in the tensorboard save dir, the best performing
        # model, according to the validation set, observed during the run
        callback2 = ModelCheckpoint(
            monitor='val_logger_loss',
            filename = args.model_name +'-{epoch:02d}-{val_logger_loss:.2f}',
            save_top_k=1
            )
        
        # Set the device to GPU zero if 0 is specified for devices
        if args.gpus == 0:
            args.gpus = [0]
        elif args.gpus == 1:
            args.gpus = [1]
        
        # Instantiate the trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger = logger,
            callbacks = [callback1, callback2],
            strategy=DDPSpawnStrategy(find_unused_parameters=False),
            log_every_n_steps = 100
            )
        trainer.fit(net, train_loader, test_loader)
      
        # Do some clean up to prevent leaks
        trainer = None
        logger = None
        train_dataset = None
        test_dataset = None
        train_loader = None
        test_loader = None
      
        print('\n-------------------------------------------')
        print('            Run Completed                   ')
        print('-------------------------------------------\n')
        
    print("\n==========\n Cross Validation Completed \n==========\n")
        
    return None
            
if __name__ == '__main__':

    # ------------
    # args
    # ------------
    
    # Instantiate the parser
    parser = ArgumentParser(add_help=False)
    
    ### Add the model/PTL specific args
    parser = SegmentationModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--num_workers', type=int, default=1)
    
    ### Dataset stuff
    parser.add_argument('--train_data_dir', type=str, default=None)
    parser.add_argument('--test_data_dir', type=str, default=None)
    parser.add_argument('--deformed_data_dir', type=str, default=None)
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--norm_stats_csv', type=str, default=None)
    parser.add_argument('--use_noise', type=str_to_bool, default=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    
    ### Training parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--decay', type=float, default=0)

    # Parse the args
    args = parser.parse_args()
   
    # Run the processing
    cli_main(args)
    