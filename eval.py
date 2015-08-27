import numpy as np
import json

import apollocaffe
from apollocaffe.layers import NumpyData, LstmUnit, Concat, InnerProduct, Filler

def evaluate_forward(net, net_config):
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

def evaluate(config):
    eval_net = apollocaffe.ApolloNet()
    # evaluate the net once to set up structure before loading parameters
    net_config = config["net"]
    evaluate_forward(eval_net, net_config)
    eval_net.load("%s_%d.h5" % (config["logging"]["snapshot_prefix"],
        config["solver"]["max_iter"] - 1))
    targets, predictions, residuals = evaluate_forward(eval_net, net_config)
    targets = ',\t'.join([('%.2f' % x) for x in targets])
    predictions = ',\t'.join([('%.2f' % x) for x in predictions])
    residuals = ',\t'.join([('%.2f' % x) for x in residuals])
    print 'The target values were:'
    print targets
    print ''
    print "The network predicted:"
    print predictions
    print ''
    print "The residuals are:"
    print residuals

def main():
    parser = apollocaffe.base_parser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    evaluate(config)

if __name__ == "__main__":
    main()
