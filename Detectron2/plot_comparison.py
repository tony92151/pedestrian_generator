import pylab
import matplotlib.pyplot as plt
import csv
import os
######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all_csv_dir', type=str, default=None)

parser.add_argument('-o','--output_dir', type=str, default=None)

parser.add_argument('-f')
args = parser.parse_args()

######################################################################################

csv_paths = args.all_csv_dir

output_dir = args.output_dir

output_path = output_dir+'/comparison.jpg'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
leg = list()

for csv_path in os.listdir(csv_paths):
    x = []
    y = []
    p = csv_paths + csv_path
    with open(p, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            x.append(float(row[0]))
            y.append(float(row[1]))
    line, = ax.plot(x ,y , 'o-')
    leg.append(csv_path[:-4])
    

ax.legend( leg, loc='lower left', shadow=True)

ax.set_xlabel('FPPI')
ax.set_ylabel('MR')

ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig(output_path)

print("Image save at :", output_path)
# plt.show()