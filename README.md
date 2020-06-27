# cycleGAN project
CentraleSupélec **cycleGAN** (2019-2020)


# How to use the JSON config file :
```javascript

{

    "output_folder" : "/workdir/output/OrangeApple_ex_1", // where to save plots and models (if a model already exists in this folder, the model will be automatically loaded and the fittig will continue)
    "on_gpu" : true, // if true : use gpu, if false : use cpu
    "number_of_epochs" : 100, //number of epoch in one pbs submission


    "dataset" : "horsesVsZebras", // name of the dataset
    "data_x_folder" : "/workdir/datas/horses/", // folder of the x datas
    "data_y_folder" : "/workdir/datas/zebras/", // folder of the y datas
    "batch_size" : 1,
    "test_ratio" : 0.2, // ratio of datas use for test
    "debug_sample" : -1, // number of example for training (if -1, use all data)


    "plot_sample" : false, // plot sample of the training dataset
    "vis_lines" : 1, // lines of the plot
    "vis_rows" : 3, // rows of the plot
    "plot_size" : 20, // size of the plot


    "save_plots" : true, // save plots at the end of each epoch
    "n_sample" : 3, // number of samples plotted 
    "save_models" : true, // if true : save the models at the end of each epoch

    "save_plots_during_batch" : true, // save plots at the end of batches 
    "freq_plots_during_batch" : 1000, // periods of saving (number of batches between two plots)


    "alpha" : 0.0002, // start learning_rate
    "starting_decay" : 100, // epochs before linear decay
    "decay_steps" : 100, // size of the support of the linear decrease
    "end_learning_rate" : 0, // learning_rate at ("starting_decay" + "decay_steps")
    "beta_1" : 0.5, // beta_1 in Adam optimizer
    "max_buffer_size" : 50, // size of buffer of generated images
    "n_resnet" : 9, // number of resnet layers

}

```
# How to run a training session :
```console
(conda_env) user@user-m:~$ python main.py config_file.json
```
