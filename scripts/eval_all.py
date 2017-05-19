import glob
import os

datacfg = os.sys.argv[1]
cfgfile = os.sys.argv[2]
backups = os.sys.argv[3]
gpu_id = os.sys.argv[4]

weights = glob.glob(backups + '*.weights')
results = []
for _, weight in enumerate(weights[:2]):
    idx = weight.split('/')[-1].split('_')[-1].split('.')[0]
    print('Evaluating %s...' % idx)
    os.system('./darknet detector valid %s %s %s -i %s -out test_%s_ > /dev/null 2> /dev/null' % (datacfg, cfgfile, weight, gpu_id, idx))
    os.system('python scripts/cars_eval.py %s results/ test_%s_ 0.5 > tmp.txt' % (datacfg, idx))
    lines = open('tmp.txt').readlines()
    if (idx == 'final'):
        idx = int(1000000000)
    else:
        idx = int(idx)
    results.append([idx, float(lines[1].split(' ')[-1]), float(lines[2].split(' ')[-1]), float(lines[3].split(' ')[-1])])
    print('%d / %d completed.' % (_ + 1, len(weights)))

results.sort(key=lambda x: x[0])

lines = ['%8d: Max Recall: %f, Max Precision: %f, mAP: %f\n' % (x[0], x[1], x[2], x[3]) for x in results[:-1]]
rec, prec, mAP = results[-1][1:]
lines.append('   final: Max Recall: %f, Max Precision: %f, mAP: %f\n' % (rec, prec, mAP))
open('results.txt', 'w').writelines(lines)
