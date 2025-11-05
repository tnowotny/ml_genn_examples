import numpy as np
import mnist
import os
import json
import matplotlib.pyplot as plt

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Checkpoint, SpikeRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Conv2D, Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                linear_latency_encode_data)

from ml_genn.compilers.event_prop_compiler import default_params

p= {"NUM_HIDDEN": 128,
    "NUM_OUTPUT": 10,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 200,
    "EXAMPLE_TIME": 20.0,
    "DT": 1.0,
    "NUM_KERNELS": 32,
    "KERNEL_SZ": 4,
    "HIDDEN_MEAN_1": 0.0,
    "HIDDEN_STD_1": 5.0,
    "READOUT": "avg_var_exp_weight",
}

TRAIN = True
KERNEL_PROFILING = True

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    p["EXAMPLE_TIME"] - (2.0 * p["DT"]), 2.0 * p["DT"])

serialiser = Numpy("latency_mnist_checkpoints")
network = SequentialNetwork(default_params)
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=p["BATCH_SIZE"] * calc_max_spikes(spikes)),
                                  (28, 28, 1), name="input", record_spikes=True)
    initial_hidden1_weight = Normal(mean=p["HIDDEN_MEAN_1"], sd=p["HIDDEN_STD_1"])
    hidden1 = Layer(Conv2D(initial_hidden1_weight, p["NUM_KERNELS"], p["KERNEL_SZ"], True),
                    LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                       tau_refrac=None),
                    synapse=Exponential(5.0), name="hidden1", record_spikes=True)
    output = Layer(Dense(Normal(mean=0.2, sd=0.37)),
                   LeakyIntegrate(tau_mem=20.0, readout=p["READOUT"]),
                   p["NUM_OUTPUT"], Exponential(5.0), name="output")

i = 0
found = True
while found:
    try:
        resfile= open(os.path.join("./", f"lmnist_conv_results{i}.txt"), "r")
    except:
        found = False
    else:
        i = i+1

with open(os.path.join("./", f"lmnist_conv{i}.json"), "w") as f:
    json.dump(p,f,indent=4)
    
fname = os.path.join("./", f"lmnist_conv_results{i}.txt")

max_example_timesteps = int(np.ceil(p["EXAMPLE_TIME"] / p["DT"]))
if TRAIN:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                 optimiser=Adam(0.5e-3), batch_size=p["BATCH_SIZE"],
                                 kernel_profiling=KERNEL_PROFILING,max_spikes=15,)
    compiled_net = compiler.compile(network)


    
    resfile= open(fname, "w")
    resfile.write(f"# Epoch ")
    resfile.write(f"hidden_n_zero1 mean_hidden_mean_spike1 std_hidden_mean_spike1 mean_hidden_std_spikes1 std_hidden_std_spikes1 val_hidden_n_zero1 val_mean_hidden_mean_spike1 val_std_hidden_mean_spike1 val_mean_hidden_std_spikes1 val_std_hidden_std_spikes1 ")
    resfile.write(f"train_accuracy validation_accuracy\n")
    resfile.close()
    with compiled_net:
        start_time = perf_counter()
        #callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        callbacks = [ SpikeRecorder(hidden1, key="spikes_hidden1",record_counts=True)]

        hidden1_conn = compiled_net.connection_populations[hidden1.connection()]
        all_g = []
        for e in range(p["NUM_EPOCHS"]):
            hidden1_conn.vars["g"].pull_from_device()
            g_view = hidden1_conn.vars["g"].view.copy()
            vmin= np.min(g_view)
            vmax= np.max(g_view)
            g_view = g_view.reshape((p["NUM_KERNELS"],p["KERNEL_SZ"],p["KERNEL_SZ"]))
            all_g.append(g_view)
            if e%10 == 0:
                fig, ax = plt.subplots(4,4)
                for i in range(4):
                    for j in range(4):
                        ax[i,j].imshow(g_view[i*4+j,:,:], vmin=vmin, vmax=vmax)
            metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train({input: spikes},
                                                                            {output: labels},
                                                                             num_epochs=1, start_epoch= e, shuffle=True,
                                                                             callbacks=callbacks, validation_callbacks=callbacks,
                                                                             validation_split = 0.1)

            print(f"Epoch: {e}, Accuracy: {metrics[output].result}, Validation Accuracy: {val_metrics[output].result}")
            resfile= open(fname, "a")
            resfile.write(f"{e} ")
            n0 = np.asarray(cb_data["spikes_hidden1"])
            mean_n0 = np.mean(n0, axis = 0)
            std_n0 = np.std(n0, axis = 0)
            n0_val = np.asarray(val_cb_data["spikes_hidden1"])
            mean_n0_val = np.mean(n0_val, axis = 0)
            std_n0_val = np.std(n0_val, axis = 0)
            resfile.write(f"{np.count_nonzero(mean_n0==0)/len(mean_n0)} {np.mean(mean_n0)} {np.std(mean_n0)} {np.mean(std_n0)} {np.std(std_n0)} {np.count_nonzero(mean_n0_val==0)/len(mean_n0_val)} {np.mean(mean_n0_val)} {np.std(mean_n0_val)} {np.mean(std_n0_val)} {np.std(std_n0_val)} ")
            resfile.write(f"{metrics[output].result} {val_metrics[output].result}\n")
            resfile.close()

        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")

        if KERNEL_PROFILING:
            print(f"Neuron update time = {compiled_net.genn_model.neuron_update_time}")
            print(f"Presynaptic update time = {compiled_net.genn_model.presynaptic_update_time}")
            print(f"Gradient batch reduce time = {compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')}")
            print(f"Gradient learn time = {compiled_net.genn_model.get_custom_update_time('GradientLearn')}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")
            print(f"Softmax1 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')}")
            print(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')}")
            print(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}")

        hidden1_conn.vars["g"].pull_from_device()
        g_view = hidden1_conn.vars["g"].view.copy()
        vmin= np.min(g_view)
        vmax= np.max(g_view)
        g_view = g_view.reshape((p["NUM_KERNELS"],p["KERNEL_SZ"],p["KERNEL_SZ"]))
        all_g.append(g_view)
        fig, ax = plt.subplots(4,4)
        for i in range(4):
            for j in range(4):
                ax[i,j].imshow(g_view[i*4+j,:,:], vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(4,4)
        for i in range(4):
            for j in range(4):
                ax[i,j].imshow(all_g[-1][i*4+j,:,:]-all_g[0][i*4+j,:,:])

        all_g = np.asarray(all_g)
        np.save("kernels.npy", all_g)
        plt.show()
        
        
            
        
else:
    # Load network state from final checkpoint
    network.load((p["NUM_EPOCHS"] - 1,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                 reset_in_syn_between_batches=True,
                                 batch_size=p["BATCH_SIZE"])
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
