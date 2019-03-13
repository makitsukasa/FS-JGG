import datetime
import numpy as np
import warnings
from jgg import JGG
from fsjgg import FSJGG
from problem.frontier.sphere      import sphere
from problem.frontier.ktablet     import ktablet
from problem.frontier.bohachevsky import bohachevsky
from problem.frontier.ackley      import ackley
from problem.frontier.schaffer    import schaffer
from problem.frontier.rastrigin   import rastrigin

warnings.simplefilter("error", RuntimeWarning)

SAVE_CSV = False

n = 20

problems = [
	{"name" : "sphere",      "func" : sphere,      "npop" :  6 * n, "nchi" : 6 * n},
	{"name" : "k-tablet",    "func" : ktablet,     "npop" :  8 * n, "nchi" : 6 * n},
	{"name" : "bohachevsky", "func" : bohachevsky, "npop" :  6 * n, "nchi" : 6 * n},
	{"name" : "ackley",      "func" : ackley,      "npop" :  8 * n, "nchi" : 6 * n},
	{"name" : "schaffer",    "func" : schaffer,    "npop" : 10 * n, "nchi" : 8 * n},
	{"name" : "rastrigin",   "func" : rastrigin,   "npop" : 24 * n, "nchi" : 8 * n},
]

datestr = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

for problem in problems:
	func = problem["func"]
	name = problem["name"]
	npop = problem["npop"]
	npar = n + 1
	nchi = problem["nchi"]
	jgg_counts = []
	bgg_barometric_counts = []
	bgg_fixed_counts = []
	eval_counts = {}
	t = 1e-2

	print(name)

	for i in range(100):
		jgg = JGG(n, npop, npar, nchi, func)
		result = jgg.until(1e-7, 300000)
		if result:
			if "JGG" in eval_counts:
				eval_counts["JGG"].append(jgg.eval_count)
			else:
				eval_counts["JGG"] = [jgg.eval_count]
		else:
			print("JGG failed")

		if SAVE_CSV:
			filename = "benchmark/{0}_jgg_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in jgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		fsjgg = FSJGG(t, n, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_not_replace
		result = fsjgg.until(1e-7, 300000)
		if result:
			if "入替無" in eval_counts:
				eval_counts["入替無"].append(fsjgg.get_eval_count())
			else:
				eval_counts["入替無"] = [fsjgg.get_eval_count()]
		else:
			print("入替無 failed")

		if SAVE_CSV:
			filename = "benchmark/{0}_入替無_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		fsjgg = FSJGG(t, n, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_rand_parents_by_elites
		result = fsjgg.until(1e-7, 300000)
		if result:
			if "親全部" in eval_counts:
				eval_counts["親全部"].append(fsjgg.get_eval_count())
			else:
				eval_counts["親全部"] = [fsjgg.get_eval_count()]
		else:
			print("親全部 failed")

		if SAVE_CSV:
			filename = "benchmark/{0}_親全部_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		fsjgg = FSJGG(t, n, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg = lambda :\
			fsjgg.choose_population_to_jgg_replace_rand_parents_by_elites(npar // 3)
		result = fsjgg.until(1e-7, 300000)
		if result:
			if "ラ親" in eval_counts:
				eval_counts["ラ親"].append(fsjgg.get_eval_count())
			else:
				eval_counts["ラ親"] = [fsjgg.get_eval_count()]
		else:
			print("ラ親 failed")

		if SAVE_CSV:
			filename = "benchmark/{0}_ラ親_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		fsjgg = FSJGG(t, n, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg = lambda :\
			fsjgg.choose_population_to_jgg_replace_losed_parents_by_elites(npar // 3)
		result = fsjgg.until(1e-7, 300000)
		if result:
			if "劣親" in eval_counts:
				eval_counts["劣親"].append(fsjgg.get_eval_count())
			else:
				eval_counts["劣親"] = [fsjgg.get_eval_count()]
		else:
			print("劣親 failed")

		if SAVE_CSV:
			filename = "benchmark/{0}_劣親_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		fsjgg = FSJGG(t, n, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_rand_by_elites
		result = fsjgg.until(1e-7, 300000)
		if result:
			if "ラ" in eval_counts:
				eval_counts["ラ"].append(fsjgg.get_eval_count())
			else:
				eval_counts["ラ"] = [fsjgg.get_eval_count()]
		else:
			print("ラ failed")

		if SAVE_CSV:
			filename = "benchmark/{0}_ラ_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		fsjgg = FSJGG(t, n, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_losed_by_elites
		result = fsjgg.until(1e-7, 300000)
		if result:
			if "劣" in eval_counts:
				eval_counts["劣"].append(fsjgg.get_eval_count())
			else:
				eval_counts["劣"] = [fsjgg.get_eval_count()]
		else:
			print("劣 failed")

		if SAVE_CSV:
			filename = "benchmark/{0}_劣_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

	for name, eval_count in eval_counts.items():
		print(name, np.average(eval_count))
