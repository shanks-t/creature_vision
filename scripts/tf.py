#!/usr/bin/env python
# tf.py
import subprocess
import sys

commands = [
    'terraform init',
    'terraform fmt',
    'terraform validate',
    'terraform plan -var-file=secrets.tfvars'
]

for command in commands:
    print(f'Running command: {command}')
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Error occurred while executing command: {command}')
        print(f'Error output: {result.stderr}')
        sys.exit(result.returncode)
    else:
        print(f'Success: {result.stdout}')