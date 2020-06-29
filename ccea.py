import numpy as np
import random
import copy


class Ccea:
    def __init__(self, p, n_inp, n_hid, n_out):
        self.population = {}
        self.fitness = np.zeros(p["pop_size"])
        self.pop_size = p["pop_size"]
        self.mut_rate = p["m_rate"]
        self.mut_chance = p["m_prob"]
        self.eps = p["epsilon"]
        self.fitness = np.zeros(self.pop_size)
        self.n_elites = p["n_elites"]  # Number of elites selected from each gen
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

        # pre_mutated_pop = copy.deepcopy(self.population)

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

        # Check pop before and after mutation to make sure mutation worked
        # identical_counter = 0
        # for pop_id in range(self.pop_size):
        #     popA = copy.deepcopy(self.population["pop{0}".format(pop_id)])
        #     popB = pre_mutated_pop["pop{0}".format(pop_id)]
        #     identical_counter += self.population_checker(popA, popB)
        # assert(identical_counter < self.pop_size)

        # After mutation any 2 populations should not be the same
        # for pop_id1 in range(self.pop_size-1):
        #     hom_counter = 0
        #     pop_id2 = pop_id1 + 1
        #     while pop_id2 < self.pop_size:
        #         assert(pop_id1 != pop_id2)
        #         pop1 = copy.deepcopy(self.population["pop{0}".format(pop_id1)])
        #         pop2 = copy.deepcopy(self.population["pop{0}".format(pop_id2)])
        #         hom_counter += self.population_checker(pop1, pop2)
        #         pop_id2 += 1
            # assert(hom_counter < self.n_elites)

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

    def fitness_prop_selection(self):
        """
        Choose next generation of policies using fitness proportional selection
        :return:
        """
        summed_fitness = np.sum(self.fitness)
        fit_brackets = np.zeros(self.pop_size)

        # Calculate fitness proportions for selections
        for pop_id in range(self.pop_size):
            if pop_id == 0:
                fit_brackets[pop_id] = self.fitness[pop_id] / summed_fitness
            else:
                fit_brackets[pop_id] = fit_brackets[pop_id - 1] + self.fitness[pop_id] / summed_fitness
        new_population = {}

        for pop_id in range(self.pop_size):
            if pop_id < self.n_elites:
                print(pop_id)
                new_population["pop{0}".format(pop_id)] = self.population["pop{0}".format(pop_id)].copy()
            else:
                rnum = random.uniform(0, 1)
                for p_id in range(self.pop_size):
                    if p_id == 0 and rnum < fit_brackets[0]:
                        print(pop_id)
                        new_population["pop{0}".format(pop_id)] = self.population["pop{0}".format(0)].copy()
                        break
                    elif fit_brackets[p_id - 1] <= rnum < fit_brackets[p_id]:
                        print(pop_id)
                        new_population["pop{0}".format(pop_id)] = self.population["pop{0}".format(p_id)].copy()
                        break

        self.population = {}
        self.population = new_population.copy()

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
        self.epsilon_greedy_select()  # Select K successors using epsilon greedy
        # self.fitness_prop_selection()  # Select k successors using fit prop selection
        self.weight_mutate()  # Mutate successors

    def reset_fitness(self):
        self.fitness = np.ones(self.pop_size)*-1
