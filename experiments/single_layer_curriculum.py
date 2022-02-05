import os
import ast
import math
import torch
import stable_nalu
import argparse
import stable_nalu.functional.regualizer as Regualizer
from decimal import Decimal

# Parse arguments
parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--layer-type',
                    action='store',
                    default='NALU',
                    choices=list(stable_nalu.network.SimpleFunctionStaticNetwork.UNIT_NAMES),
                    type=str,
                    help='Specify the layer type, e.g. Tanh, ReLU, NAC, NALU')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=[
                        'add', 'sub', 'mul', 'div', 'squared', 'root'
                    ],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul, squared')
parser.add_argument('--num-subsets',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the number of subsets to use')
parser.add_argument('--regualizer',
                    action='store',
                    default=10,
                    type=float,
                    help='Specify the regualization lambda to be used')
parser.add_argument('--regualizer-z',
                    action='store',
                    default=0,
                    type=float,
                    help='Specify the z-regualization lambda to be used')
parser.add_argument('--regualizer-oob',
                    action='store',
                    default=1,
                    type=float,
                    help='Specify the oob-regualization lambda to be used')
parser.add_argument('--first-layer',
                    action='store',
                    default=None,
                    help='Set the first layer to be a different type')

parser.add_argument('--max-iterations',
                    action='store',
                    default=100000,
                    type=int,
                    help='Specify the max number of iterations to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=128,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')

parser.add_argument('--interpolation-range',
                    action='store',
                    default=[1,2],
                    type=ast.literal_eval,
                    help='Specify the interpolation range that is sampled uniformly from')
parser.add_argument('--extrapolation-range',
                    action='store',
                    default=[2,6],
                    type=ast.literal_eval,
                    help='Specify the extrapolation range that is sampled uniformly from')
parser.add_argument('--input-size',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the input size')
parser.add_argument('--subset-ratio',
                    action='store',
                    default=0.5,
                    type=float,
                    help='Specify the subset-size as a fraction of the input-size')
parser.add_argument('--overlap-ratio',
                    action='store',
                    default=0.0,
                    type=float,
                    help='Specify the overlap-size as a fraction of the input-size')
parser.add_argument('--simple',
                    action='store_true',
                    default=False,
                    help='Use a very simple dataset with t = sum(v[0:2]) + sum(v[4:6])')

parser.add_argument('--hidden-size',
                    action='store',
                    default=2,
                    type=int,
                    help='Specify the vector size of the hidden layer.')
parser.add_argument('--nac-mul',
                    action='store',
                    default='none',
                    choices=['none', 'normal', 'safe', 'max-safe', 'mnac', 'npu', 'real-npu'],
                    type=str,
                    help='Make the second NAC a multiplicative NAC, used in case of a just NAC network.')
parser.add_argument('--oob-mode',
                    action='store',
                    default='clip',
                    choices=['regualized', 'clip'],
                    type=str,
                    help='Choose of out-of-bound should be handled by clipping or regualization.')
parser.add_argument('--regualizer-scaling',
                    action='store',
                    default='linear',
                    choices=['exp', 'linear'],
                    type=str,
                    help='Use an expoentational scaling from 0 to 1, or a linear scaling.')
parser.add_argument('--regualizer-scaling-start',
                    action='store',
                    default=1000000,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=2000000,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer-shape',
                    action='store',
                    default='linear',
                    choices=['squared', 'linear', 'none'],
                    type=str,
                    help='Use either a squared or linear shape for the bias and oob regualizer. Use none so W reg in tensorboard is logged at 0')
parser.add_argument('--mnac-epsilon',
                    action='store',
                    default=0,
                    type=float,
                    help='Set the idendity epsilon for MNAC.')
parser.add_argument('--nalu-bias',
                    action='store_true',
                    default=False,
                    help='Enables bias in the NALU gate')
parser.add_argument('--nalu-two-nac',
                    action='store_true',
                    default=False,
                    help='Uses two independent NACs in the NALU Layer')
parser.add_argument('--nalu-two-gate',
                    action='store_true',
                    default=False,
                    help='Uses two independent gates in the NALU Layer')
parser.add_argument('--nalu-mul',
                    action='store',
                    default='normal',
                    choices=['normal', 'safe', 'trig', 'max-safe', 'mnac'],
                    help='Multplication unit, can be normal, safe, trig')
parser.add_argument('--nalu-gate',
                    action='store',
                    default='normal',
                    choices=['normal', 'regualized', 'obs-gumbel', 'gumbel'],
                    type=str,
                    help='Can be normal, regualized, obs-gumbel, or gumbel')

parser.add_argument('--optimizer',
                    action='store',
                    default='adam',
                    choices=['adam', 'sgd'],
                    type=str,
                    help='The optimization algorithm to use, Adam or SGD')
parser.add_argument('--learning-rate',
                    action='store',
                    default=1e-3,
                    type=float,
                    help='Specify the learning-rate')
parser.add_argument('--momentum',
                    action='store',
                    default=0.0,
                    type=float,
                    help='Specify the nestrov momentum, only used with SGD')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='simple_function_static',
                    type=str,
                    help='Where the data should be stored')
parser.add_argument('--remove-existing-data',
                    action='store_true',
                    default=False,
                    help='Should old results be removed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')

parser.add_argument('--regualizer-beta-start',
                    action='store',
                    default=1e-5,
                    type=float,
                    help='Starting value of the beta scale factor.')
parser.add_argument('--regualizer-beta-end',
                    action='store',
                    default=1e-4,
                    type=float,
                    help='Final value of the beta scale factor.')
parser.add_argument('--regualizer-beta-step',
                    action='store',
                    default=10000,
                    type=int,
                    help='Update the regualizer-beta-start value every x steps.')
parser.add_argument('--regualizer-beta-growth',
                    action='store',
                    default=10,
                    type=int,
                    help='Scale factor to grow the regualizer-beta-start value by.')
parser.add_argument('--regualizer-l1',
                    action='store_true',
                    default=False,
                    help='Add L1 regularization loss term. Be sure the regualizer-scaling is set')
parser.add_argument('--regualizer-npu-w',
                    action='store_true',
                    default=False,
                    help='Add L1 regularization loss term. Be sure the regualizer-scaling is set')
parser.add_argument('--regualizer-gate',
                    type=int,
                    default=0,
                    help='')

parser.add_argument('--pytorch-precision',
                    type=int,
                    default=32,
                    help='Precision for pytorch to work in')
                    
# TODO - add fixed gate option?

args = parser.parse_args()

if args.pytorch_precision == 32:
  torch.set_default_dtype(torch.float32)
elif args.pytorch_precision == 64:
  torch.set_default_dtype(torch.float64)
else:
  raise ValueError(f'Unsupported pytorch_precision option ({args.pytorch_precision})')

setattr(args, 'cuda', torch.cuda.is_available() and not args.no_cuda)

# Print configuration
print(f'running')
print(f'  - layer_type: {args.layer_type}')
print(f'  - first_layer: {args.first_layer}')
print(f'  - operation: {args.operation}')
print(f'  - num_subsets: {args.num_subsets}')
print(f'  - regualizer: {args.regualizer}')
print(f'  - regualizer_z: {args.regualizer_z}')
print(f'  - regualizer_oob: {args.regualizer_oob}')
print(f'  -')
print(f'  - max_iterations: {args.max_iterations}')
print(f'  - batch_size: {args.batch_size}')
print(f'  - seed: {args.seed}')
print(f'  -')
print(f'  - interpolation_range: {args.interpolation_range}')
print(f'  - extrapolation_range: {args.extrapolation_range}')
print(f'  - input_size: {args.input_size}')
print(f'  - subset_ratio: {args.subset_ratio}')
print(f'  - overlap_ratio: {args.overlap_ratio}')
print(f'  - simple: {args.simple}')
print(f'  -')
print(f'  - hidden_size: {args.hidden_size}')
print(f'  - nac_mul: {args.nac_mul}')
print(f'  - oob_mode: {args.oob_mode}')
print(f'  - regualizer_scaling: {args.regualizer_scaling}')
print(f'  - regualizer_scaling_start: {args.regualizer_scaling_start}')
print(f'  - regualizer_scaling_end: {args.regualizer_scaling_end}')
print(f'  - regualizer_shape: {args.regualizer_shape}')
print(f'  - mnac_epsilon: {args.mnac_epsilon}')
print(f'  - nalu_bias: {args.nalu_bias}')
print(f'  - nalu_two_nac: {args.nalu_two_nac}')
print(f'  - nalu_two_gate: {args.nalu_two_gate}')
print(f'  - nalu_mul: {args.nalu_mul}')
print(f'  - nalu_gate: {args.nalu_gate}')
print(f'  -')
print(f'  - optimizer: {args.optimizer}')
print(f'  - learning_rate: {args.learning_rate}')
print(f'  - momentum: {args.momentum}')
print(f'  -')
print(f'  - cuda: {args.cuda}')
print(f'  - name_prefix: {args.name_prefix}')
print(f'  - remove_existing_data: {args.remove_existing_data}')
print(f'  - verbose: {args.verbose}')
print(f'  -')
print(f'  - regualizer_beta_start: {args.regualizer_beta_start}')
print(f'  - regualizer_beta_end: {args.regualizer_beta_end}')
print(f'  - regualizer_beta_step: {args.regualizer_beta_step}')
print(f'  - regualizer_beta_growth: {args.regualizer_beta_growth}')
print(f'  - regualizer_l1: {args.regualizer_l1}')
print(f'  - regualizer-npu-w: {args.regualizer_npu_w}')
print(f'  - regualizer-gate: {args.regualizer_gate}')
print(f'  - pytorch-precision: {torch.get_default_dtype()}')

# Prepear logging
summary_writer = stable_nalu.writer.SummaryWriter(
    f'{args.name_prefix}/{args.layer_type.lower()}'
    # f'{"-nac-" if args.nac_mul != "none" else ""}'
    # f'{"n" if args.nac_mul == "normal" else ""}'
    # f'{"s" if args.nac_mul == "safe" else ""}'
    # f'{"s" if args.nac_mul == "max-safe" else ""}'
    # f'{"t" if args.nac_mul == "trig" else ""}'
    # f'{"m" if args.nac_mul == "mnac" else ""}'
    # f'{"npu" if args.nac_mul == "npu" else ""}'
    # f'{"npur" if args.nac_mul == "real-npu" else ""}'
    # f'{"-nalu-" if (args.nalu_bias or args.nalu_two_nac or args.nalu_two_gate or args.nalu_mul != "normal" or args.nalu_gate != "normal") else ""}'
    f'{"-b" if args.nalu_bias and args.layer_type == "NALU" else ""}'
    f'{"-2n" if args.nalu_two_nac and args.layer_type == "NALU" else ""}'
    f'{"-2g" if args.nalu_two_gate and args.layer_type == "NALU" else ""}'
    f'{"-s" if args.nalu_mul == "safe" and args.layer_type == "NALU" else ""}'
    f'{"-s" if args.nalu_mul == "max-safe" and args.layer_type == "NALU" else ""}'
    f'{"-t" if args.nalu_mul == "trig" and args.layer_type == "NALU" else ""}'
    f'{"-m" if args.nalu_mul == "mnac" and args.layer_type == "NALU" else ""}'
    f'{"-r" if args.nalu_gate == "regualized" and args.layer_type == "NALU" else ""}'       # TODO - move npu reg args here?
    f'{"-u" if args.nalu_gate == "gumbel" and args.layer_type == "NALU" else ""}'
    f'{"-uu" if args.nalu_gate == "obs-gumbel" and args.layer_type == "NALU" else ""}'
    f'_op-{args.operation.lower()}'
    f'_oob-{"c" if args.oob_mode == "clip" else "r"}'
    f'_rs-{args.regualizer_scaling}-{args.regualizer_shape}'
    f'_eps-{args.mnac_epsilon}'
    f'_rl-{args.regualizer_scaling_start}-{args.regualizer_scaling_end}'
    f'_r-{args.regualizer}-{args.regualizer_z}-{args.regualizer_oob}'
    f'_i-{args.interpolation_range[0]}-{args.interpolation_range[1]}'
    f'_e-{args.extrapolation_range[0]}-{args.extrapolation_range[1]}'
    f'_z-{"simple" if args.simple else f"{args.input_size}-{args.subset_ratio}-{args.overlap_ratio}"}'
    f'_b{args.batch_size}'
    f'_s{args.seed}'
    f'_h{args.hidden_size}'
    f'_z{args.num_subsets}'
    f'_lr-{args.optimizer}-{"%.5f" % args.learning_rate}-{args.momentum}'
    f'_L1{"T" if args.regualizer_l1 else f"F"}'
    f'_rb-{args.regualizer_beta_start}-{args.regualizer_beta_end}-{args.regualizer_beta_step}-{args.regualizer_beta_growth}'
    f'_rWnpu{"T" if args.regualizer_npu_w else f"F"}'
    f'_rg-{args.regualizer_gate}'
    f'_p-{args.pytorch_precision}',
    remove_existing_data=args.remove_existing_data
)

# Set threads
if 'LSB_DJOB_NUMPROC' in os.environ:
    torch.set_num_threads(int(os.environ['LSB_DJOB_NUMPROC']))

# Set seed
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Setup datasets
dataset = stable_nalu.dataset.SimpleFunctionStaticDataset(
    operation=args.operation,
    input_size=args.input_size,
    subset_ratio=args.subset_ratio,
    overlap_ratio=args.overlap_ratio,
    num_subsets=args.num_subsets,
    simple=args.simple,
    use_cuda=args.cuda,
    seed=args.seed,
)
print(f'  -')
print(f'  - dataset: {dataset.print_operation()}')
# Interpolation and extrapolation seeds are from random.org
# expo midpoint growth
#ds1 = iter(dataset.fork(sample_range=[1.1,1.2]).dataloader(batch_size=args.batch_size))
#ds2 = iter(dataset.fork(sample_range=[1.125,1.175]).dataloader(batch_size=args.batch_size))
#ds3 = iter(dataset.fork(sample_range=[1.1375,1.1625]).dataloader(batch_size=args.batch_size))
#ds4 = iter(dataset.fork(sample_range=[1.14375,1.14375]).dataloader(batch_size=args.batch_size))

# fixed by difference
#ds1 = iter(dataset.fork(sample_range=[1.1,1.2]).dataloader(batch_size=args.batch_size))
#ds2 = iter(dataset.fork(sample_range=[1.11,1.19]).dataloader(batch_size=args.batch_size))
#ds3 = iter(dataset.fork(sample_range=[1.12,1.18]).dataloader(batch_size=args.batch_size))
#ds4 = iter(dataset.fork(sample_range=[1.13,1.17]).dataloader(batch_size=args.batch_size))

# LtoR linear sweep
ds1 = iter(dataset.fork(sample_range=[1.1,1.125]).dataloader(batch_size=args.batch_size))
ds2 = iter(dataset.fork(sample_range=[1.1,1.15]).dataloader(batch_size=args.batch_size))
ds3 = iter(dataset.fork(sample_range=[1.1,1.175]).dataloader(batch_size=args.batch_size))
ds4 = iter(dataset.fork(sample_range=[1.1,1.2]).dataloader(batch_size=args.batch_size))

# RtoL sweep
#ds1 = iter(dataset.fork(sample_range=[1.175,1.2]).dataloader(batch_size=args.batch_size))
#ds2 = iter(dataset.fork(sample_range=[1.15,1.2]).dataloader(batch_size=args.batch_size))
#ds3 = iter(dataset.fork(sample_range=[1.125,1.2]).dataloader(batch_size=args.batch_size))
#ds4 = iter(dataset.fork(sample_range=[1.1,1.2]).dataloader(batch_size=args.batch_size))

# linear midpoint growth
#ds1 = iter(dataset.fork(sample_range=[1.1,1.2]).dataloader(batch_size=args.batch_size))
#ds2 = iter(dataset.fork(sample_range=[1.125,1.175]).dataloader(batch_size=args.batch_size))
#ds3 = iter(dataset.fork(sample_range=[1.133333,1.166667]).dataloader(batch_size=args.batch_size))
#ds4 = iter(dataset.fork(sample_range=[1.1375,1.1625]).dataloader(batch_size=args.batch_size))

dataset_train = [ds1, ds2, ds3, ds4]
dataset_valid_interpolation_data = next(iter(dataset.fork(sample_range=args.interpolation_range, seed=43953907).dataloader(batch_size=10000)))
dataset_test_extrapolation_data = next(iter(dataset.fork(sample_range=args.extrapolation_range, seed=8689336).dataloader(batch_size=10000)))

# setup model
model = stable_nalu.network.SingleLayerNetwork(
    args.layer_type,
    input_size=dataset.get_input_size(),
    writer=summary_writer.every(1000).verbose(args.verbose),
    first_layer=args.first_layer,
    hidden_size=args.hidden_size,
    nac_oob=args.oob_mode,
    regualizer_shape=args.regualizer_shape,
    regualizer_z=args.regualizer_z,
    mnac_epsilon=args.mnac_epsilon,
    nac_mul=args.nac_mul,
    nalu_bias=args.nalu_bias,
    nalu_two_nac=args.nalu_two_nac,
    nalu_two_gate=args.nalu_two_gate,
    nalu_mul=args.nalu_mul,
    nalu_gate=args.nalu_gate,
    fixed_gate=False,       # TODO - create arg flag?
    regualizer_gate=args.regualizer_gate,
    regualizer_npu_w=args.regualizer_npu_w,


)
model.reset_parameters()
if args.cuda:
    model.cuda()
criterion = torch.nn.MSELoss()

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    raise ValueError(f'{args.optimizer} is not a valid optimizer algorithm')

def test_model(data):
    with torch.no_grad(), model.no_internal_logging(), model.no_random():
        x, t = data
        return criterion(model(x), t)

# Train model
print(model)
print('')
print(summary_writer.name)
print('')

use_npu_scaling = args.regualizer_l1 or args.regualizer_npu_w or args.regualizer_gate
if use_npu_scaling:
    # Decimal type required to avoid accumulation of fp precision errors when multiplying by growth factor
    args.regualizer_beta_start = Decimal(str(args.regualizer_beta_start))
    # Decimal and fp arithmetic don't mix so beta end must also be a decimal
    args.regualizer_beta_end = Decimal(str(args.regualizer_beta_end))
r_l1_scale = args.regualizer_beta_start

train_ds_idx = 0
# TODO - make args 
cl_thr = 1e-2
cl_growth = 0.1
epoch_i = 0
while epoch_i < args.max_iterations+1:
    (x_train, t_train) = next(dataset_train[train_ds_idx])
    summary_writer.set_iteration(epoch_i)

    # Prepear model
    model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
    optimizer.zero_grad()

    # Log validation
    if epoch_i % 1000 == 0:
        interpolation_error = test_model(dataset_valid_interpolation_data)
        extrapolation_error = test_model(dataset_test_extrapolation_data)

        summary_writer.add_scalar('metric/valid/interpolation', interpolation_error)
        summary_writer.add_scalar('metric/test/extrapolation', extrapolation_error)

    # forward
    y_train = model(x_train)
    regualizers = model.regualizer()    # logs 3 reg metrics to tensorbord if verbose

    if (args.regualizer_scaling == 'linear'):
        r_w_scale = max(0, min(1, (
            (epoch_i - args.regualizer_scaling_start) /
            (args.regualizer_scaling_end - args.regualizer_scaling_start)
        )))
    elif (args.regualizer_scaling == 'exp'):
        r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

    l1_loss = 0
    if args.regualizer_l1:
        l1_loss = Regualizer.l1(model.parameters())
        if args.verbose:
            summary_writer.add_scalar('L1/train/L1-loss', l1_loss)

    if use_npu_scaling:
        # the beta_start value will be updated accordingly to be the correct beta value for the epoch.
        # It is done this way to avoid having initalise another variable outside the epoch loop
        if epoch_i % args.regualizer_beta_step == 0 and epoch_i != 0:
            if args.regualizer_beta_start < args.regualizer_beta_end:
                args.regualizer_beta_start *= args.regualizer_beta_growth

        r_l1_scale = float(args.regualizer_beta_start)  # Decimal doesn't work for tensorboard or mixed fp arithmetic
        summary_writer.add_scalar('L1/train/beta', r_l1_scale)

    loss_train_criterion = criterion(y_train, t_train)

    loss_train_regualizer = args.regualizer * r_w_scale * regualizers['W'] + \
                            regualizers['g'] + \
                            args.regualizer_z * regualizers['z'] + \
                            args.regualizer_oob * regualizers['W-OOB'] + \
                            int(args.regualizer_l1) * r_l1_scale * l1_loss + \
                            int(args.regualizer_npu_w) * r_l1_scale * regualizers['W-NPU'] + \
                            args.regualizer_gate * r_l1_scale * regualizers['g-NPU']

    loss_train = loss_train_criterion + loss_train_regualizer

    # Log loss
    if args.verbose or epoch_i % 1000 == 0:
        summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
        summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
        summary_writer.add_scalar('loss/train/total', loss_train)
    
    if epoch_i % 1000 == 0:
        print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

    # Optimize model
    if loss_train.requires_grad:
        loss_train.backward()
        optimizer.step()
    model.optimize(loss_train_criterion)

    # Log gradients if in verbose mode
    if args.verbose and epoch_i % 1000 == 0:
        model.log_gradients()
    
    # move to next curriculum
    if loss_train < cl_thr and train_ds_idx < len(dataset_train)-1:
      #print(f'epoch {epoch_i}, ds: {train_ds_idx}, x_train {x_train}, cl_thr {cl_thr}, train loss: {loss_train}')
      train_ds_idx += 1
      cl_thr *= cl_growth
      print(f'cl update to ds {train_ds_idx}')
    
    epoch_i += 1

# Compute validation loss
loss_valid_inter = test_model(dataset_valid_interpolation_data)
loss_valid_extra = test_model(dataset_test_extrapolation_data)

# Write results for this training
print(f'finished:')
print(f'  - loss_train: {loss_train}')
print(f'  - loss_valid_inter: {loss_valid_inter}')
print(f'  - loss_valid_extra: {loss_valid_extra}')

model.writer._root.close()  # fix - close summary writer before saving model to avoid thread locking issues

# Use saved weights to visualize the intermediate values.
#stable_nalu.writer.save_model(summary_writer.name, model)
# stable_nalu.writer.save_model('../save/test', model)  #uncommment to save on windows
