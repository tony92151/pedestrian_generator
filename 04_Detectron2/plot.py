import pylab
import matplotlib.pyplot as plt
import csv

######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default=None)

parser.add_argument('--output_path', type=str, default=None)

parser.add_argument('-f')
args = parser.parse_args()

######################################################################################

csv_path = args.csv_path

output_dir = args.output_path

output_path = output_dir

results = []
x = []
y = []

with open(csv_path, newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        tmp = map(float,row)
        results.append(list(tmp))
        x.append(float(row[0]))
        y.append(float(row[1]))
        
print(results)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

line = ax.plot(x ,y , 'o-')

ax.set_xlabel('FPPI')
ax.set_ylabel('MR')

ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig(output_path)

print("Image save at :", output_path)
# plt.show()