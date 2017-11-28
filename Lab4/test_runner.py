#!/usr/bin/env python

import subprocess
import itertools
import time


def main():
    input_sizes = [32, 64, 128, 256, 512, 1024]
    filter_sizes = [4, 8, 16, 32, 64]

    size_combinations = filter(
        lambda (i, f): i >= f,
        itertools.product(input_sizes, filter_sizes)
    )

    cmds = ['cpu', 'gpu']

    base_cmd = './build/convolution_{}'
    #cmd_str = './build/convolution_{0} {1} {1} {2} {2}'

    results = {}
    for idx, (i, f) in enumerate(size_combinations):
        results[(i, f)] = {}
        
        for c in cmds:
            times = []
            print "For %dx%d on %s:" % (i, f, c),
            for _ in range(10):
                cmd = [base_cmd.format(c)] + [str(i)]*2 + [str(f)]*2 + ['--no-write']
            
                begin = time.time()
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = p.communicate()
                end = time.time()
                
                times.append(end - begin)
                
                print " %0.6f" % (end - begin),
            
            average = sum(times) / len(times)
            
            print "; avg =", average
            
            results[(i, f)][c] = average

    print results
    
    for c in cmds:
        print ',' + ','.join([str(x) for x in filter_sizes])
        for i in input_sizes: 
            print "%d,%s" % (i, ','.join(["%0.6f" % results[(i, f)][c] for f in filter_sizes if (i, f) in results]))
            """for f in filter_sizes:
                d = results.get((i, f), {})
                print ",%0.6f" % d.get(c, -1.0),
            print """""
        print ""

if __name__ == '__main__':
    main()
