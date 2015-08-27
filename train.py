import numpy as np
import json
import random

import apollocaffe
from apollocaffe.layers import (NumpyData, LstmUnit, Concat, InnerProduct,
    EuclideanLoss, Filler)

def lstm_layers(net, step, filler, net_config):
    layer_list = []
    if step == 0:
        prev_hidden = "lstm_seed"
        prev_mem = "lstm_seed"
    else:
        prev_hidden = "lstm_hidden%d" % (step - 1)
        prev_mem = "lstm_mem%d" % (step - 1)
    # Concatenate the hidden output with the next input value
    layer_list.append(Concat("lstm_concat%d" % step,
        bottoms=[prev_hidden, "value%d" % step]))
    # Run the LSTM for one more step
    layer_list.append(LstmUnit("lstm%d" % step, net_config["mem_cells"],
        bottoms=["lstm_concat%d" % step, prev_mem],
        param_names=["input_value", "input_gate", "forget_gate", "output_gate"],
        tops=["lstm_hidden%d" % step, "lstm_mem%d" % step],
        weight_filler=filler))
    return layer_list

def forward(net, net_config):
    net.clear_forward()
    length = random.randrange(net_config["min_len"], net_config["max_len"])

    # initialize all weights in [-0.1, 0.1]
    filler = Filler("uniform", net_config["init_range"])
    # initialize the LSTM memory with all 0's
    net.f(NumpyData("lstm_seed",
        np.zeros((net_config["batch_size"], net_config["mem_cells"]))))
    accum = np.zeros((net_config["batch_size"],))

    # Begin recurrence through 5 - 15 inputs
    for step in range(length):
        # Generate random inputs
        value = np.array([random.random()
            for _ in range(net_config["batch_size"])])
        # Set data of value blob to contain a batch of random numbers
        net.f(NumpyData("value%d" % step,
            value.reshape((-1, 1))))
        accum += value
        for l in lstm_layers(net, step, filler, net_config):
            net.f(l)

    # Add a fully connected layer with a bottom blob set to be the last used
    # LSTM cell. Note that the network structure is now a function of the data
    net.f(InnerProduct("ip", 1, bottoms=["lstm_hidden%d" % (length - 1)],
        weight_filler=filler))
    # Add a label for the sum of the inputs
    net.f(NumpyData("label", np.reshape(accum, (-1, 1))))
    # Compute the Euclidean loss between the preiction and label,
    # used for backprop
    net.f(EuclideanLoss("euclidean", bottoms=["ip", "label"]))

def train(config):
    net = apollocaffe.ApolloNet()

    net_config = config["net"]
    solver = config["solver"]
    logging = config["logging"]
    loggers = [
        apollocaffe.loggers.TrainLogger(logging["display_interval"]),
        apollocaffe.loggers.SnapshotLogger(
            logging["snapshot_interval"], logging["snapshot_prefix"]),
        ]
    train_loss = []
    for i in range(solver["max_iter"]):
        forward(net, net_config)
        train_loss.append(net.loss)
        net.backward()
        lr = (solver["base_lr"] * (solver["gamma"])**(i // solver["stepsize"]))
        net.update(lr=lr, momentum=solver["momentum"],
            clip_gradients=solver["clip_gradients"])
        for logger in loggers:
            logger.log(i, {"train_loss": train_loss, "apollo_net": net})

def main():
    parser = apollocaffe.base_parser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    train(config)

if __name__ == "__main__":
    main()
