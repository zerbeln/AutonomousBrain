import numpy as np
import random
import copy
from parameters import population_size, mutation_rate, mutation_prob, epsilon, num_elites


class Ccea:
    def __init__(self, n_inp, n_hid, n_out):
        self.population = {}
        self.fitness = np.zeros(population_size)
        self.pop_size = population_size
        self.mut_rate = mutation_rate
        self.mut_chance = mutation_prob
        self.eps = epsilon
        self.fitness = np.zeros(self.pop_size)
        self.n_elites = num_elites  # Number of elites selected from each gen
        self.team_selection = np.ones(self.pop_size) * (-1)

        # Network Parameters that determine the number of weights to evolve
        self.n_inputs = n_inp
        self.n_outputs = n_out
        self.n_hidden = n_hid

    def create_new_population(self):  # Re-initializes CCEA populations for new run
        """
        Create new populations (for beginning of stat run)
        :return: None
        """
        self.population = {}
        self.fitness = np.zeros(self.pop_size)
        self.team_selection = np.ones(self.pop_size) * (-1)

        for pop_id in range(self.pop_size):
            policy = {}
            policy["L1"] = np.random.normal(0, 0.5, self.n_inputs * self.n_hidden)
            policy["L2"] = np.random.normal(0, 0.5, self.n_hidden * self.n_outputs)
            policy["b1"] = np.random.normal(0, 0.5, self.n_hidden)
            policy["b2"] = np.random.normal(0, 0.5, self.n_outputs)

            self.population["pop{0}".format(pop_id)] = policy.copy()

    def select_policy_teams(self):  # Create policy teams for testing
        """
        Choose teams of individuals from among populations to be tested
        :return: None
        """

        self.team_selection = random.sample(range(self.pop_size), self.pop_size)

    def weight_mutate(self):
        """
        Mutate offspring populations (each weight has a probability of mutation)
        :return:
        """
        pop_id = int(self.n_elites)
        while pop_id < self.pop_size:
            mut_counter = 0
            # First Weight Layer
            for w in range(self.n_inputs*self.n_hidden):
                rnum1 = random.uniform(0, 1)
                if rnum1 <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["L1"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop{0}".format(pop_id)]["L1"][w] += mutation

            # Second Weight Layer
            for w in range(self.n_hidden*self.n_outputs):
                rnum2 = random.uniform(0, 1)
                if rnum2 <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["L2"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop{0}".format(pop_id)]["L2"][w] += mutation

            # Output bias weights
            for w in range(self.n_hidden):
                rnum3 = random.uniform(0, 1)
                if rnum3 <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["b1"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop{0}".format(pop_id)]["b1"][w] += mutation

            # Output layer weights
            for w in range(self.n_outputs):
                rnum4 = random.uniform(0, 1)
                if rnum4 <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["b2"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop{0}".format(pop_id)]["b2"][w] += mutation

            pop_id += 1

    def population_checker(self, pop1, pop2):
        matrix_count = 0

        # Layer 1 Weights
        l1_counter = 0
        for w in range(self.n_inputs*self.n_hidden):
            if pop1["L1"][w] == pop2["L1"][w]:
                l1_counter += 1
        if l1_counter == self.n_inputs*self.n_hidden:
            matrix_count += 1

        # Layer 2 Weights
        l2_counter = 0
        for w in range(self.n_hidden*self.n_outputs):
            if pop1["L2"][w] == pop2["L2"][w]:
                l2_counter += 1
        if l2_counter == self.n_hidden*self.n_outputs:
            matrix_count += 1

        # Bias 1 Weights
        b1_counter = 0
        for w in range(self.n_hidden):
            if pop1["b1"][w] == pop2["b1"][w]:
                b1_counter += 1
        if b1_counter == self.n_hidden:
            matrix_count += 1

        # Bias 2 Weights
        b2_counter = 0
        for w in range(self.n_outputs):
            if pop1["b2"][w] == pop2["b2"][w]:
                b2_counter += 1
        if b2_counter == self.n_outputs:
            matrix_count += 1

        if matrix_count == 4:
            return 1
        else:
            return 0

    def binary_tournament_selection(self):
        """
        Select parents using binary tournament selection
        :return:
        """
        new_population = {}
        for pop_id in range(self.pop_size):
            if pop_id < self.n_elites:
                new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(pop_id)])
            else:
                p1 = random.randint(0, self.pop_size-1)
                p2 = random.randint(0, self.pop_size-1)
                while p1 == p2:
                    p2 = random.randint(0, self.pop_size - 1)

                if self.fitness[p1] > self.fitness[p2]:
                    new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(p1)])
                elif self.fitness[p1] < self.fitness[p2]:
                    new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(p2)])
                else:
                    rnum = random.uniform(0, 1)
                    if rnum > 0.5:
                        new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(p1)])
                    else:
                        new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(p2)])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents using e-greedy selection
        :return: None
        """
        new_population = {}
        for pop_id in range(self.pop_size):
            if pop_id < self.n_elites:
                new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(pop_id)])
            else:
                rnum = random.uniform(0, 1)
                if rnum < self.eps:
                    max_index = np.argmax(self.fitness)
                    new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(max_index)])
                else:
                    parent = random.randint(1, (self.pop_size - 1))
                    new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(parent)])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def random_selection(self):
        """
        Choose next generation of policies using elite-random selection
        :return:
        """
        new_population = {}
        for pop_id in range(self.pop_size):
            if pop_id < self.n_elites:
                new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(pop_id)])
            else:
                parent = random.randint(0, self.pop_size-1)
                new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(parent)])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def rank_population(self):
        """
        Reorders the population in terms of fitness (high to low)
        :return:
        """
        ranked_population = copy.deepcopy(self.population)
        for pop_id_a in range(self.pop_size):
            pop_id_b = pop_id_a + 1
            ranked_population["pop{0}".format(pop_id_a)] = copy.deepcopy(self.population["pop{0}".format(pop_id_a)])
            while pop_id_b < (self.pop_size):
                if pop_id_a != pop_id_b:
                    if self.fitness[pop_id_a] < self.fitness[pop_id_b]:
                        self.fitness[pop_id_a], self.fitness[pop_id_b] = self.fitness[pop_id_b], self.fitness[pop_id_a]
                        ranked_population["pop{0}".format(pop_id_a)] = copy.deepcopy(self.population["pop{0}".format(pop_id_b)])
                pop_id_b += 1

        self.population = {}
        self.population = copy.deepcopy(ranked_population)

    def down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parents create offspring population, and perform mutation operations
        :return: None
        """
        self.rank_population()
        # self.epsilon_greedy_select()  # Select K successors using epsilon greedy
        self.binary_tournament_selection()
        # self.random_selection()  # Select k successors using fit prop selection
        self.weight_mutate()  # Mutate successors

    def reset_fitness(self):
        """
        Clear fitness of non-elite policies
        :return:
        """

        pol_id = self.n_elites
        while pol_id < self.pop_size:
            self.fitness[pol_id] = 0.00
            pol_id += 1
