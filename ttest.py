import sys
import csv
from scipy import stats

def ttest(filenames, log_scaled = False):
	if isinstance(filenames, str):
		print("give 2 or more files")
		exit(-1)

	datas = {}

	for filename in filenames:
		with open(filename, "r") as f:
			x = []
			y = []
			reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC)
			datas[filename] = [x for x in reader]

	for i in range(len(filenames)):
		for j in range(len(filenames)):
			result = stats.ttest_ind(datas[filenames[i]], datas[filenames[j]])
			filename_i = filenames[i].split("\\")[-1].split(".")[0]
			filename_j = filenames[j].split("\\")[-1].split(".")[0]
			print(filename_i, filename_j,
				"有意差:", "あり" if result.pvalue < 0.05 else "なし",
				"t: {0:.3f} p: {1:.3f}".format(result.statistic[0], result.pvalue[0]))

if __name__ == '__main__':
	if len(sys.argv) > 1:
		ttest(sys.argv[1:])
	else:
		ttest([
			"benchmark/JGG.csv",
			"benchmark/親全部.csv",
		])
