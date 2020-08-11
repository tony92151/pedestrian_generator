import pylab
import matplotlib.pyplot as plt
import csv

######################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default=None)

parser.add_argument('--output_dir', type=str, default=None)

parser.add_argument('-f')
args = parser.parse_args()

######################################################################################

csv_path = args.csv_path

output_dir = args.output_dir

output_path = output_dir+'/plot.jpg'

results = []
x = []
y = []
with open(csv_path, newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        tmp = map(float,row)
        results.append(list(tmp))
print(results)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

line = ax.plot(results, 'b', lw=2)

ax.set_xscale('log')
plt.savefig(output_path)

print("Image save at :", output_path)
# plt.show()