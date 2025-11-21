1. `CNN_module.py`: define CNN architecture

2. `datasets.py`: `create train_dataset, val_dataset, test_dataset, device`

3. `train.py`: normal training script

4. `hyperparameter_search.py`: for model-specific (such as convolution channels, filter size etc.) and training-specific (such as learning rate,random seeds) hyperparameters
    python hyperparameter_search.py
        <-- internally search for architecture-specific parameters:
        check results in: ./results/architecture_hyperparam_search_20251121_163147.csv

        df = pd.read_csv("./results/architecture_hyperparam_search_20251121_163147.csv")
        # the combination gives the best validation accuracy:
        df.loc[df['final_val_acc'].idxmax()]

            conv_channels                                                  [16, 32]
            kernel_sizes                                                          5
            activation                                                         relu
            fc_layer_sizes                                                 [64, 15]
            dropout                                                             0.4
            train_loss_history    [2.124961985179356, 1.0660606152216594, 0.6631...
            train_acc_history     [0.30333333333333334, 0.6381904761904762, 0.77...
            val_loss_history      [1.2626976863013373, 0.5716620660887825, 0.405...
            val_acc_history       [0.6275555555555555, 0.8271111111111111, 0.879...
            best_val_acc                                                   0.974222
            best_epoch                                                           27
            final_train_acc                                                0.971048
            final_val_acc                                                  0.973778
            training_config       {'batch_size': 32, 'epochs': 30, 'learning_rat...
            success                                                            True
            Name: 0, dtype: object
        

5. `test.py`: model performance on the testing for classification task and plot the confusion matrix
    python test.py results/model_config_20251121_162932.json results/weights_20251121_162932.pt

6. 'historty_plot.py': plot history
    python history_plot.py results/training_history_20251121_162932.csv
