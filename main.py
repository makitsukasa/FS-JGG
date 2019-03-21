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

SAVE_HISTORY_CSV = False
SAVE_COUNTS_CSV = False

N = 20
GOAL = 1e-7
MAX_EVAL_COUNT = 2**18
T = 1e-2
LOOP_COUNT = 3

PROBLEMS = [
	{"name" : "sphere",      "func" : sphere,      "npop" :  6 * N, "nchi" : 6 * N},
	{"name" : "k-tablet",    "func" : ktablet,     "npop" : 10 * N, "nchi" : 6 * N},
	{"name" : "bohachevsky", "func" : bohachevsky, "npop" :  8 * N, "nchi" : 6 * N},
	{"name" : "ackley",      "func" : ackley,      "npop" :  8 * N, "nchi" : 6 * N},
	# {"name" : "schaffer",    "func" : schaffer,    "npop" : 11 * N, "nchi" : 8 * N},
	# {"name" : "rastrigin",   "func" : rastrigin,   "npop" : 24 * N, "nchi" : 8 * N},
]

datestr = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

for problem in PROBLEMS:
	func = problem["func"]
	name = problem["name"]
	npop = problem["npop"]
	npar = N + 1
	nchi = problem["nchi"]
	eval_counts = {}

	print(name, LOOP_COUNT, flush = True)

	for i in range(LOOP_COUNT):
		np.random.seed()
		randseed = np.random.randint(0x7fffffff)

		np.random.seed(randseed)
		jgg = JGG(N, npop, npar, nchi, func)
		result = jgg.until(GOAL, MAX_EVAL_COUNT)
		if result:
			if "JGG" in eval_counts:
				eval_counts["JGG"].append(jgg.eval_count)
			else:
				eval_counts["JGG"] = [jgg.eval_count]
		else:
			print("JGG failed")

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_jgg_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in jgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		np.random.seed(randseed)
		fsjgg = FSJGG(T, N, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_not_replace
		result = fsjgg.until(GOAL, MAX_EVAL_COUNT)
		if result:
			if "入替無" in eval_counts:
				eval_counts["入替無"].append(fsjgg.get_eval_count())
			else:
				eval_counts["入替無"] = [fsjgg.get_eval_count()]
		else:
			print("入替無 failed", randseed)

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_入替無_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		np.random.seed(randseed)
		fsjgg = FSJGG(T, N, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_rand_parents_by_elites
		result = fsjgg.until(GOAL, MAX_EVAL_COUNT)
		if result:
			if "親全部" in eval_counts:
				eval_counts["親全部"].append(fsjgg.get_eval_count())
			else:
				eval_counts["親全部"] = [fsjgg.get_eval_count()]
		else:
			print("親全部 failed", randseed)

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_親全部_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		np.random.seed(randseed)
		fsjgg = FSJGG(T, N, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_rand_by_elites
		result = fsjgg.until(GOAL, MAX_EVAL_COUNT)
		if result:
			if "ラ" in eval_counts:
				eval_counts["ラ"].append(fsjgg.get_eval_count())
			else:
				eval_counts["ラ"] = [fsjgg.get_eval_count()]
		else:
			print("ラ failed", randseed)

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_ラ_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		np.random.seed(randseed)
		fsjgg = FSJGG(T, N, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_losed_by_elites
		result = fsjgg.until(GOAL, MAX_EVAL_COUNT)
		if result:
			if "劣" in eval_counts:
				eval_counts["劣"].append(fsjgg.get_eval_count())
			else:
				eval_counts["劣"] = [fsjgg.get_eval_count()]
		else:
			print("劣 failed", randseed)

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_劣_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

	for method_name, eval_count in eval_counts.items():
		print(method_name, np.average(eval_count), LOOP_COUNT - len(eval_count))

		if SAVE_COUNTS_CSV:
			filename = "benchmark/{0}_{1}_{2}.csv".format(datestr, name, method_name)
			with open(filename, "w") as f:
				for c in eval_count:
					f.write("{}\n".format(c))
				f.close()
