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
	# {"name" : "sphere",      "func" : sphere,      "npop" :  6 * n, "nchi" : 6 * n},
	{"name" : "k-tablet",    "func" : ktablet,     "npop" :  8 * n, "nchi" : 6 * n},
	# {"name" : "bohachevsky", "func" : bohachevsky, "npop" :  6 * n, "nchi" : 6 * n},
	# {"name" : "ackley",      "func" : ackley,      "npop" :  8 * n, "nchi" : 6 * n},
	# {"name" : "schaffer",    "func" : schaffer,    "npop" : 10 * n, "nchi" : 8 * n},
	# {"name" : "rastrigin",   "func" : rastrigin,   "npop" : 24 * n, "nchi" : 8 * n},
]

datestr = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

for problem in problems:
	func = problem["func"]
	name = problem["name"]
	npop = problem["npop"]
	npar = n + 1
	nchi = problem["nchi"]
	eval_counts = {}
	t = 1e-2
	loop_count = 10000

	print(name, loop_count, flush = True)

	for i in range(loop_count):
		randseed = 1401383486
		# randseed = 1651120371
		# randseed = 1727460444

		np.random.seed(randseed)
		fsjgg = FSJGG(t, n, npop, npar, nchi, func)
		np.random.seed()
		fsjgg.choose_population_to_jgg =\
			fsjgg.choose_population_to_jgg_replace_rand_parents_by_elites
		result = fsjgg.until(1e-7, 300000)
		if result:
			if "親全部" in eval_counts:
				eval_counts["親全部"].append(fsjgg.get_eval_count())
			else:
				eval_counts["親全部"] = [fsjgg.get_eval_count()]
		else:
			print("親全部 failed", randseed)

		if SAVE_CSV:
			filename = "benchmark/{0}_親全部_{1}_{2}.csv".format(datestr, name, i)
			with open(filename, "w") as f:
				for c, v in fsjgg.history.items():
					f.write("{0},{1}\n".format(c, v))
				f.close()

	for name, eval_count in eval_counts.items():
		print(name, np.average(eval_count), loop_count - len(eval_count))
