<div id="LSTM" class="subgroup">
<h4>LSTM</h4>
<p> With the basics laid out, let's see a more complicated example that showcases some of Apollocaffe's strengths.
We'll construct an LSTM network that learns to sum up a variable length sequence of 5-15 inputs distributed uniformly in [0,1].
Best practices dictate that we separate out the hyperparameters into a separate <a href="https://github.com/Russell91/LSTMSummation/blob/master/config.json">config.json</a> file that doesn't mingle with the rest of the code.
</p>

<p>
Next, we'll define the forward pass of the network in it's own function.
</p>

<div class="highlight">
<pre><code class="language-python" data-lang="python">def lstm_layers(net, step, filler, net_config):
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
</code></pre>
<p>
On each forward pass, we first randomly determine the length of the network. At each step,
a new LSTM block outputs two tops - one for the hidden state and one for the memory.
Subsequent LSTM blocks concatenates the hidden output with the new random value.
When the network is finished unrolling, we add an InnerProduct layer to calculate the sum from the final LSTM unit,
and train with a EuclideanLoss layer comparing this value to the accumulated sum calculated in python.
<br/>
<br/>
Next we'll set up the training loop as before. But this time, we'll add a few bells and whistles
like network snapshots to HDF5 files, automatic plotting of the training loss, and better logging.
</p>
<pre><code class="language-python" data-lang="python">def train(config):
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
</code></pre>
<p>
We can begin training with:
<pre><code>python train.py --gpu -1 --config config.json </code></pre>
</p>
<h4>Evaluation</h4>
<p>
After we're done training, we'll want to evaluate the network's performance. To keep things interesting,
we'll see if the network is able to <i>generalise</i> to sequences of longer length than it was trained on. We
set the recurrent length to 20 numbers, all having value 0.5. Hopefully the network will output 10!
</p>
<pre><code class="language-python" data-lang="python">def evaluate_forward(net, net_config):
    net.clear_forward()
    length = 20

    net.f(NumpyData("prev_hidden", np.zeros((1, net_config["mem_cells"]))))
    net.f(NumpyData("prev_mem", np.zeros((1, net_config["mem_cells"]))))
    filler = Filler("uniform", net_config["init_range"])
    predictions = []

    value = 0.5
    for _ in range(length):
        # We'll be updating values in place for efficient memory usage. This
        # will break backprop and cause warnings. Use clear_forward to suppress.
        net.clear_forward()

        # Add 0.5 to the sum at each step
        net.f(NumpyData("value",
            data=np.array(value).reshape((1, 1))))
        prev_hidden = "prev_hidden"
        prev_mem = "prev_mem"
        net.f(Concat("lstm_concat", bottoms=[prev_hidden, "value"]))
        net.f(LstmUnit("lstm", net_config["mem_cells"],
            bottoms=["lstm_concat", prev_mem],
            param_names=[
                "input_value", "input_gate", "forget_gate", "output_gate"],
            weight_filler=filler,
            tops=["next_hidden", "next_mem"]))
        net.f(InnerProduct("ip", 1, bottoms=["next_hidden"]))
        predictions.append(float(net.blobs["ip"].data.flatten()[0]))
        # set up for next prediction by copying LSTM outputs back to inputs
        net.blobs["prev_hidden"].data_tensor.copy_from(
            net.blobs["next_hidden"].data_tensor)
        net.blobs["prev_mem"].data_tensor.copy_from(
            net.blobs["next_mem"].data_tensor)

    targets = np.cumsum([value for _ in predictions])
    residuals = [x - y for x, y in zip(predictions, targets)]
    return targets, predictions, residuals
</code></pre>
    <p>
    We'll use a distinct net for evaluation. First, we seed the LSTM with 0's as before, but this time
    we copy the hidden state and memory output of the LSTM block back to it's input at each iteration.
    So far, when accessing blob data, we've used the .data property, which copies the data
    to the cpu and returns a numpy array referencing the data. This is quite convenient, as numpy is more
    fully featured and familiar than the Tensor library. But in this case, we don't want to pay the overhead
    of transferring data back and forth from the CPU to the GPU.
    Instead, we use the .data_tensor property, which returns a reference to the underlying tensor. We can then
    copy the data from the 'next_hidden' top to the 'prev_hidden' top directly on the GPU!
    <br/>
    <br/>
    Now we are finally ready to run the network. We and run the code with:
</p>
            <pre><code>python eval.py --gpu -1 --config config.json </code></pre>
<p>
    The output is: <br/>
    <pre><code>2015-08-27 14:33:19 - CPU device
The target values were:
0.50,   1.00,   1.50,   2.00,   2.50,   3.00,   3.50,   4.00,   4.50,   5.00,   5.50,   6.00,   6.50,   7.00,   7.50, 8.00,    8.50,   9.00,   9.50,   10.00

The network predicted:
0.65,   1.11,   1.58,   2.05,   2.53,   3.03,   3.54,   4.07,   4.61,   5.17,   5.75,   6.34,   6.94,   7.54,   8.14, 8.74,    9.33,   9.91,   10.46,  10.99

The residuals are:
0.15,   0.11,   0.08,   0.05,   0.03,   0.03,   0.04,   0.07,   0.11,   0.17,   0.25,   0.34,   0.44,   0.54,   0.64, 0.74,    0.83,   0.91,   0.96,   0.99</pre></code>
</p>
<p>
    Recall that the network was trained on input sequencies ranging from 5 to 15 values long.
    After receiving 15 input values of 0.5, the network predicts a sum of 7.54. The cumulative error is only 0.04.
    Notably, the error is greater during the first 5 inputs.
    After 20 steps,
    the error is 0.99, but the network has achieved some level of generalization. If you run the code yourself,
    you can see what happens at even greater length.
</p>
</div>
