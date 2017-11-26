#!/usr/bin/env python

import os
import sys
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

    for i, f in size_combinations:
        begin = time.time()
        # Make call here
        end = time.time()
        
        df.loc[i, f] = end - begin


if __name__ == '__main__':
    main()

