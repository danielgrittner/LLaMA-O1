import os
table = os.popen('squeue -u zhangdi1').read()
for line in str(table).split('\n'):
    # print(line)
    if 'batch_run' in line or 'offline' in line:
        id = line.split('AI4')[0] #change to your slurm partition name
        os.system(f'scancel {id}')
# print(table)

os.system('squeue -u zhangdi1')
os.system('rm *.out')