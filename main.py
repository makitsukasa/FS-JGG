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

SAVE_HISTORY_CSV = True
SAVE_COUNTS_CSV = False

N = 20
GOAL = 1e-7
MAX_EVAL_COUNT = 50 * N
T = 1e-2
LOOP_COUNT = 1

PROBLEMS = [
	{"name" : "sphere",      "func" : sphere,      "npop" :  6 * N, "nchi" : 6 * N},
	# {"name" : "k-tablet",    "func" : ktablet,     "npop" : 10 * N, "nchi" : 6 * N},
	# {"name" : "bohachevsky", "func" : bohachevsky, "npop" :  8 * N, "nchi" : 6 * N},
	# {"name" : "ackley",      "func" : ackley,      "npop" :  8 * N, "nchi" : 6 * N},
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
	best_fitnesses = {}

	print(name, LOOP_COUNT, flush = True)

	for i in range(LOOP_COUNT):
		np.random.seed()
		randseed = np.random.randint(0x7fffffff)

		np.random.seed(randseed)
		jgg = JGG(N, npop, npar, nchi, func)
		result = jgg.until(GOAL, MAX_EVAL_COUNT)
		if "JGG" in best_fitnesses:
			best_fitnesses["JGG"].append(jgg.get_best_fitness())
		else:
			best_fitnesses["JGG"] = [jgg.get_best_fitness()]

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
		if "入替無" in best_fitnesses:
			best_fitnesses["入替無"].append(fsjgg.get_best_fitness())
		else:
			best_fitnesses["入替無"] = [fsjgg.get_best_fitness()]

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_入替無_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.jgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		np.random.seed(randseed)
		fsjgg = FSJGG(T, N, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_rand_parents_by_elites
		result = fsjgg.until(GOAL, MAX_EVAL_COUNT)
		if "親全部" in best_fitnesses:
			best_fitnesses["親全部"].append(fsjgg.get_best_fitness())
		else:
			best_fitnesses["親全部"] = [fsjgg.get_best_fitness()]

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_親全部_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.jgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		np.random.seed(randseed)
		fsjgg = FSJGG(T, N, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_rand_by_elites
		result = fsjgg.until(GOAL, MAX_EVAL_COUNT)
		if "ラ" in best_fitnesses:
			best_fitnesses["ラ"].append(fsjgg.get_best_fitness())
		else:
			best_fitnesses["ラ"] = [fsjgg.get_best_fitness()]

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_ラ_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.jgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

		np.random.seed(randseed)
		fsjgg = FSJGG(T, N, npop, npar, nchi, func)
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_losed_by_elites
		result = fsjgg.until(GOAL, MAX_EVAL_COUNT)
		if "劣" in best_fitnesses:
			best_fitnesses["劣"].append(fsjgg.get_best_fitness())
		else:
			best_fitnesses["劣"] = [fsjgg.get_best_fitness()]

		if SAVE_HISTORY_CSV:
			filename = "benchmark/{0}_劣_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.jgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

	for method_name, eval_count in best_fitnesses.items():
		print(method_name, np.average(eval_count), LOOP_COUNT - len(eval_count))

		if SAVE_COUNTS_CSV:
			filename = "benchmark/{0}_{1}_{2}.csv".format(datestr, name, method_name)
			with open(filename, "w") as f:
				for c in eval_count:
					f.write("{}\n".format(c))
				f.close()
