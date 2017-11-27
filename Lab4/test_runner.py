#!/usr/bin/env python

import subprocess
import itertools
import time
import pandas as pd


def main():
    input_sizes = [32, 64, 128, 256, 512, 1024]
    filter_sizes = [4, 8, 16, 32, 64]

    size_combinations = filter(
        lambda (i, f): i >= f,
        itertools.product(input_sizes, filter_sizes)
    )

    df = pd.DataFrame(index=input_sizes, columns=filter_sizes)

    cmds = ['cpu', 'gpu']

    cmd_str = './convolution_{0} {1} {1} {2} {2}'

    for i, f in size_combinations:
        for c in cmds:
            begin = time.time()

            p = subprocess.Popen(
                cmd_str.format(c, i, f),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            out, err = p.communicate()

            end = time.time()

            df.loc[i, f] = end - begin


if __name__ == '__main__':
    main()
