import re
import ast
import pandas
import collections
import multiprocessing

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from .tensorboard_reader import TensorboardReader


def _parse_numpy_str(array_string):
    pattern = r'''# Match (mandatory) whitespace between...
                (?<=\]) # ] and
                \s+
                (?= \[) # [, or
                |
                (?<=[^\[\]\s])
                \s+
                (?= [^\[\]\s]) # two non-bracket non-whitespace characters
            '''

    # Replace such whitespace with a comma
    fixed_string = re.sub(pattern, ',', array_string, flags=re.VERBOSE)
    return np.array(ast.literal_eval(fixed_string))


def _csv_format_column_name(column_name):
    return column_name.replace('/', '.')


def _everything_default_matcher(tag):
    return True


def norm_and_percent(val, state, name):
    s0_train_dp, s0_test_dp = 72000, 9000
    not_s0_train_dp, not_s0_test_dp = 73000, 8000
    
    if "_f0_" in name:
        n = s0_train_dp if state == 'train' else s0_test_dp
    else:
        n = not_s0_train_dp if state == 'train' else not_s0_test_dp
    return 100 * val / n


class TensorboardMetricTwoDigitMNISTReader:
    def __init__(self, dirname,
                 metric_matcher=_everything_default_matcher,
                 step_start=0,
                 processes=None,
                 progress_bar=True):
        self.dirname = dirname
        self.metric_matcher = metric_matcher
        self.step_start = step_start

        self.processes = processes
        self.progress_bar = progress_bar

    def _parse_tensorboard_data(self, inputs):
        (dirname, filename, reader) = inputs

        columns = collections.defaultdict(list)
        columns['name'] = dirname

        current_epoch = None
        current_logged_step = None

        for e in tf.compat.v1.train.summary_iterator(filename):
            step = e.step - self.step_start

            for v in e.summary.value:
                if v.tag == 'epoch':
                    current_epoch = v.simple_value

                elif self.metric_matcher(v.tag):
                    columns[v.tag].append(v.simple_value)
                    current_logged_step = step

                    # Syncronize the step count with the loss metrics
                    if len(columns['step']) != len(columns[v.tag]):
                        columns['step'].append(step)

                    # Syncronize the wall.time with the loss metrics
                    # if len(columns['wall.time']) != len(columns[v.tag]):
                    #     columns['wall.time'].append(e.wall_time)

                    # Syncronize the epoch with the loss metrics
                    if current_epoch is not None and len(columns['epoch']) != len(columns[v.tag]):
                        columns['epoch'].append(current_epoch)
                        
                # deal with separately since not all models will have these logged
                elif v.tag.endswith('label2out/weights/w0') and current_logged_step == step:
                    if len(columns['step']) != len(columns['label2out.w0']):
                        columns['label2out.w0'].append(v.simple_value)
                elif v.tag.endswith('label2out/weights/w1') and current_logged_step == step:
                    if len(columns['step']) != len(columns['label2out.w1']):
                        columns['label2out.w1'].append(v.simple_value)

                # elif v.tag.endswith('W/sparsity_error') and current_logged_step == step:
                #     # Step changed, update sparse error
                #     if len(columns['step']) != len(columns['sparse.error.max']):
                #         columns['sparse.error.max'].append(v.simple_value)
                #     else:
                #         columns['sparse.error.max'][-1] = max(
                #             columns['sparse.error.max'][-1],
                #             v.simple_value
                #         )

                # FIXME - ONLY USE THIS CODE TO COMPENSATE FOR A PREVIOUS BUG!
                # deal with normalising and converting rat eto percentage for the 4dp and 4dp acc metrics
                # if v.tag.endswith('dp/acc') and current_logged_step == step:
                #     val = norm_and_percent(v.simple_value, state=v.tag.split('/')[1], name=columns['name'])
                #
                #     if v.tag.endswith('train/output_rounded_4dp/acc'):
                #         if len(columns['step']) == len(columns['metric/train/output_rounded_4dp/acc']):
                #             columns['metric/train/output_rounded_4dp/acc'][-1] = val
                #
                #     elif v.tag.endswith('train/output_rounded_5dp/acc'):
                #         if len(columns['step']) == len(columns['metric/train/output_rounded_5dp/acc']):
                #             columns['metric/train/output_rounded_5dp/acc'][-1] = val
                #
                #     elif v.tag.endswith('test/output_rounded_4dp/acc'):
                #         if len(columns['step']) == len(columns['metric/test/output_rounded_4dp/acc']):
                #             columns['metric/test/output_rounded_4dp/acc'][-1] = val
                #
                #     elif v.tag.endswith('test/output_rounded_5dp/acc'):
                #         if len(columns['step']) == len(columns['metric/test/output_rounded_5dp/acc']):
                #             columns['metric/test/output_rounded_5dp/acc'][-1] = val

        # if len(columns['sparse.error.max']) == 0:
        #     columns['sparse.error.max'] = [None] * len(columns['step'])

        # deal with cases where no label2out weights are logged
        if len(columns['label2out.w0']) == 0:
            columns['label2out.w0'] = [None] * len(columns['step'])
        if len(columns['label2out.w1']) == 0:
            columns['label2out.w1'] = [None] * len(columns['step'])
        return dirname, columns

    def __iter__(self):
        reader = TensorboardReader(self.dirname, auto_open=False)
        with tqdm(total=len(reader), disable=not self.progress_bar) as pbar, \
                multiprocessing.Pool(self.processes) as pool:

            columns_order = None
            for dirname, data in pool.imap_unordered(self._parse_tensorboard_data, reader):
                pbar.update()

                # Check that some data is present
                # if len(data['step']) == 0:
                #     print(f'missing data in: {dirname}')
                #     continue

                # Fix flushing issue
                for column_name, column_data in data.items():
                    if len(data['step']) - len(column_data) == 1:
                        data[column_name].append(None)

                # Convert to dataframe
                df = pandas.DataFrame(data)
                if len(df) == 0:
                    print(f'Warning: No data for {dirname}')
                    continue

                # Ensure the columns are always order the same
                if columns_order is None:
                    columns_order = df.columns.tolist()
                else:
                    df = df[columns_order]

                df.rename(_csv_format_column_name, axis='columns', inplace=True)
                yield df
