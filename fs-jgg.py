from copy import deepcopy
import datetime
import numpy as np
from individual import Individual
import plot
from jgg import JGG
from fs import FS

class FSJGG(object):
	def __init__(self, t, n, npop, npar, nchi, problem):
		self.npar = npar
		self.fs = FS(t, n, npop, npar, nchi, problem)
		self.jgg = JGG(n, npop, npar, nchi, problem)

		self.init_population = self.fs.population

	def inherit(self):
		self.jgg.history = deepcopy(self.fs.history)
		self.jgg.eval_count = self.fs.eval_count
		self.jgg.population = self.choose_population_to_jgg()

	def until(self, goal = 1e-7, max_count = 300000):
		result = self.fs.until_stuck(max_count)
		if not result:
			return False
		self.inherit()
		return self.jgg.until(goal, max_count)

	def get_eval_count(self):
		return max(self.fs.eval_count, self.jgg.eval_count)

	def get_best_fitness(self):
		return min(self.fs.get_best_fitness(), self.jgg.get_best_fitness())

	def choose_population_to_jgg_replace_rand_parents_by_elites(self, elites_count = None):
		if elites_count is None:
			elites_count = self.npar
		ret = self.init_population[:]
		np.random.shuffle(ret)
		ret = ret[elites_count:]
		elites = self.fs.population[:]
		elites.sort(key = lambda i: i.fitness)
		ret.extend(elites[:elites_count])
		return ret

if __name__ == '__main__':
	n = 20
	ga = FSJGG(1e-4, n, 6 * n, n + 1, 6 * n, lambda x: np.sum((x * 10.24 - 5.12) ** 2))

	ga.choose_population_to_jgg = ga.choose_population_to_jgg_replace_rand_parents_by_elites
	ga.until(1e-7, 300000)
	print(ga.get_best_fitness(), ga.get_eval_count())

	filename = "benchmark/{0:%Y-%m-%d_%H-%M-%S}.csv".format(datetime.datetime.now())
	with open(filename, "w") as f:
		for c, v in ga.jgg.history.items():
			f.write("{0},{1}\n".format(c, v))
		f.close()

	plot.plot(filename)
