#!/usr/bin/env python3
"""Opt-in integration regression. Requires a model and two available CUDA GPUs."""
import argparse
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--miner', required=True)
p.add_argument('--model', required=True)
p.add_argument('--devices', default='0,1')
p.add_argument('--mem', default='1400')
a = p.parse_args()
devices = a.devices.split(',')
if len(set(devices)) < 2:
    p.error('select at least two distinct CUDA devices')
base = [a.miner, '--model', a.model, '--mem', a.mem]
cases = [
    ('graph, sequential devices', devices, ['--check', '--batch', '4']),
    ('eager, reversed devices', devices[::-1], ['--check', '--batch', '1', '--no-graph']),
    # Disable fused attention so this independently covers the GEMM opt-in,
    # including the 80 KiB tall tile used with deep-K planar operands.
    ('GEMM opt-in', devices, ['--check', '--batch', '1', '--attn-cublas', '--no-fused-loader', '--gemm-maxk', '999999']),
    ('concurrent devices and lanes', devices, ['--bench', '32', '--batch', '4', '--lanes', '2']),
]
for name, order, args in cases:
    print(name, flush=True)
    result = subprocess.run(base + ['--devices', ','.join(order)] + args,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    print(result.stdout, end='', flush=True)
    if result.returncode or re.search(r'FATAL|CUDA error|bench error|CHECK FAILED|FAIL:', result.stdout):
        raise SystemExit('FAIL: ' + name)
    if '--check' in args:
        if result.stdout.count('CHECK OK') != len(order):
            raise SystemExit('FAIL: not every selected GPU checked the consensus vector')
    else:
        for dev in order:
            if not re.search(r'GPU ' + re.escape(dev) + r': 256 inferences in ', result.stdout):
                raise SystemExit('FAIL: missing benchmark work on GPU ' + dev)
print('Multi-GPU regression PASS', flush=True)
