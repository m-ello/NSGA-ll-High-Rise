import numpy as np
import random


class NSGA2:
    def __init__(self, pop_size=100, gen=200, xprob=0.9, mutprob=0.1, cost_m3=135):
        self.population_size = pop_size
        self.generations = gen
        self.x_prob = xprob
        self.mutation_prob = mutprob
        self.population = None
        self.objectives = None
        self.cost_m3 = cost_m3

    def initialize_population(self, bounds):
        pop = []
        for _ in range(self.population_size):
            individual = []
            for i in range(len(bounds)):
                # 0.2 chance extreme values for diversity
                if random.random() < 0.2:
                    individual.append(bounds[i][0 if random.random() < 0.5 else 1])
                else:
                    individual.append(random.uniform(bounds[i][0], bounds[i][1]))
            pop.append(individual)
        self.population = np.array(pop)

    def evaluate_objectives(self, population):
        obj = []
        for ind in population:
            w, l, h, t = ind

            # 1. Cost: volume * cost/m³
            wall_area = 2 * (w + l) * h
            volume = wall_area * t
            cost = volume * self.cost_m3

            # 2. penalty for wall thickenss under 0.3m for instability reason
            if t < 0.3:
                cost_multiplier = 1.0 + (0.3 - t) * 3  # +60% pentru t=0.1
                cost *= cost_multiplier

            # Opțiune 1: Factor de stabilitate bazat pe raport de aspect
            stability = (w * l * t) / (h ** 1.5)

            # Opțiune 3: Simplu și intuitiv - minimizăm h/(w*t)
            # stability = h / (w * t)

            # we want to minimize both: cost and instability
            obj.append([cost, -stability])  # negative stability = instability

        return np.array(obj)

    def is_feasible(self, individual):
        """feasability test"""
        w, l, h, t = individual

        # 1. Height-to-width aspect ratio must be bigger than 8
        if h / w > 8:
            return False

        # 2. Minimal thickness for high buildings
        if h > 200 and t < 0.4:
            return False

        # 3. Reasonable length-to-width ratio
        if l / w > 3 or w / l > 3:
            return False

        return True

    def fast_nondominated_sort(self, population, objectives):
        fronts = [[]]
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]

        for i in range(len(population)):
            for j in range(len(population)):
                if i == j:
                    continue
                # minimizam ambele obiective
                if all(objectives[i] <= objectives[j]) and any(objectives[i] < objectives[j]):
                    dominated_solutions[i].append(j)
                elif all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []
            for i in fronts[front_index]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_index += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance_assignment(self, front, objectives):
        distances = {i: 0.0 for i in front}
        num_objectives = objectives.shape[1]

        for m in range(num_objectives):
            sorted_front = sorted(front, key=lambda i: objectives[i][m])
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')
            min_obj = objectives[sorted_front[0]][m]
            max_obj = objectives[sorted_front[-1]][m]
            if max_obj == min_obj:
                continue
            for k in range(1, len(sorted_front) - 1):
                i = sorted_front[k]
                prev_i = sorted_front[k - 1]
                next_i = sorted_front[k + 1]
                distances[i] += (
                                        objectives[next_i][m] - objectives[prev_i][m]
                                ) / (max_obj - min_obj)

        return [distances[i] for i in front]

    def select_parents_by_rank_and_distance(self, fronts):
        parents = []
        for front in fronts:
            if len(parents) + len(front) <= self.population_size:
                parents.extend(front)
            else:
                distances = self.crowding_distance_assignment(front, self.objectives)
                sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                remaining = self.population_size - len(parents)
                parents.extend([ind for ind, _ in sorted_front[:remaining]])
                break
        return np.array([self.population[i] for i in parents])

    def crossover_sbx(self, p1, p2, bounds):
        """Simulated Binary Crossover"""
        child1, child2 = [], []
        eta_c = 20  # Distribution index

        for j in range(len(p1)):
            if random.random() < 0.5:
                # Ensure parents are different
                if abs(p1[j] - p2[j]) > 1e-10:
                    if p1[j] < p2[j]:
                        y1, y2 = p1[j], p2[j]
                    else:
                        y1, y2 = p2[j], p1[j]

                    rand = random.random()
                    beta = 1.0 + (2.0 * (y1 - bounds[j][0]) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta_c + 1)

                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))

                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                    # Pol boundary check
                    c1 = min(max(c1, bounds[j][0]), bounds[j][1])
                    c2 = min(max(c2, bounds[j][0]), bounds[j][1])

                    if random.random() < 0.5:
                        child1.append(c1)
                        child2.append(c2)
                    else:
                        child1.append(c2)
                        child2.append(c1)
                else:
                    child1.append(p1[j])
                    child2.append(p2[j])
            else:
                child1.append(p1[j])
                child2.append(p2[j])

        return child1, child2

    def mutation_polynomial(self, child, bounds):
        """Polynomial Mutation"""
        eta_m = 20  # Mutation distribution index

        for j in range(len(child)):
            if random.random() < self.mutation_prob / len(child):
                y = child[j]
                yl, yu = bounds[j][0], bounds[j][1]

                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)

                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow

                y = y + delta_q * (yu - yl)
                child[j] = min(max(y, yl), yu)

        return child

    def solve(self, bounds):
        random.seed(42)  # for reproductibility
        np.random.seed(42)

        self.initialize_population(bounds)

        for gen in range(self.generations):
            # Evaluare obiective
            self.objectives = self.evaluate_objectives(self.population)

            # Sort Pareto fronts
            fronts = self.fast_nondominated_sort(self.population, self.objectives)

            # Display progress
            if gen % 20 == 0:
                print(f"Generation {gen}: Front Pareto = {len(fronts[0])} solutions")
                print(f"  Cost: [{np.min(self.objectives[:, 0]):.0f}, {np.max(self.objectives[:, 0]):.0f}]")
                print(f"  Stability: [{np.min(-self.objectives[:, 1]):.3f}, {np.max(-self.objectives[:, 1]):.3f}]")

            if gen == self.generations - 1:
                break

            # Select parents
            parents = self.select_parents_by_rank_and_distance(fronts)

            # Generate children
            children = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    p1, p2 = parents[i], parents[i + 1]
                    if random.random() < self.x_prob:
                        c1, c2 = self.crossover_sbx(p1, p2, bounds)
                        c1 = self.mutation_polynomial(c1, bounds)
                        c2 = self.mutation_polynomial(c2, bounds)
                        children.extend([c1, c2])
                    else:
                        children.extend([p1, p2])
                else:
                    children.append(parents[i])

            # Combine populations
            combined_pop = np.vstack([self.population, children])
            combined_obj = self.evaluate_objectives(combined_pop)

            # Sort the combination
            combined_fronts = self.fast_nondominated_sort(combined_pop, combined_obj)

            # Select new population
            new_pop = []
            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.population_size:
                    new_pop.extend(front)
                else:
                    distances = self.crowding_distance_assignment(front, combined_obj)
                    sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                    remaining = self.population_size - len(new_pop)
                    new_pop.extend([ind for ind, _ in sorted_front[:remaining]])
                    break

            self.population = np.array([combined_pop[i] for i in new_pop])

        # Final results
        final_objectives = self.evaluate_objectives(self.population)
        fronts = self.fast_nondominated_sort(self.population, final_objectives)

        pareto_front = [self.population[i] for i in fronts[0]]
        pareto_objectives = [final_objectives[i] for i in fronts[0]]

        # Sort based on cost for clearer display
        sorted_idx = np.argsort([obj[0] for obj in pareto_objectives])
        pareto_front = [pareto_front[i] for i in sorted_idx]
        pareto_objectives = [pareto_objectives[i] for i in sorted_idx]

        return pareto_front, pareto_objectives


if __name__ == "__main__":
    BOUNDS = np.array([
        [20, 60],  # w - width
        [20, 100],  # l - length
        [60, 300],  # h - height
        [0.2, 1.0],  # t - walls thickness
    ])

    nsga = NSGA2(pop_size=100, gen=100)
    pareto_solutions, pareto_obj = nsga.solve(BOUNDS)

    print("\n" + "=" * 80)
    print("FINAL PARETO SOLUTIONS (15 best):")
    print("=" * 80)

    for i, (sol, obj) in enumerate(zip(pareto_solutions[:15], pareto_obj[:15])):
        print(f"\nSolution {i + 1}:")
        print(f"  Width: {sol[0]:6.2f}m, Length: {sol[1]:6.2f}m")
        print(f"  Height: {sol[2]:6.2f}m, Thickness: {sol[3]:6.2f}m")
        print(f"  Cost: ${obj[0]:11,.2f}")
        print(f"  Stability: {-obj[1]:6.3f} (bigger = more stable)")
        print(f"  H/W ratio: {sol[2] / sol[0]:6.2f}")
