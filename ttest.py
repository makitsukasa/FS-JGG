import sys
import csv
from scipy import stats

def ttest(filenames, log_scaled = False):
	if len(filenames) != 2:
		print("give 2 files")
		exit(-1)

	datas = {}

	for filename in filenames:
		print(filename, end = " ")
		with open(filename, "r") as f:
			x = []
			y = []
			reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC)
			datas[filename] = [x for x in reader]
	print()

	result = stats.ttest_ind(datas[filenames[0]], datas[filenames[1]])
	print(result)
	if result.pvalue < 0.025:
		print("有意差あり")
	else:
		print("有意差なし")

if __name__ == '__main__':
	if len(sys.argv) > 1:
		ttest(sys.argv[1:])
	else:
		ttest([
			"benchmark/JGG.csv",
			"benchmark/親全部.csv",
		])
