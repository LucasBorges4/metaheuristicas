# Module reutilizável contendo a implementação da meta‑heurística evolutiva.
# Extraído de MetaHeuristica_Microquimerismo.ipynb.
# Inclui a classe EvolutionaryDanceEP, a leitura de instâncias e a função de experimento.

import os
import random
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


class EvolutionaryDanceEP:
    def __init__(self, instance,
                 pop_size=200,
                 mutation_rate=0.2,
                 alpha=0.31,
                 max_iter=1000,

                 status_decay_rate=0.05,
                 low_status_threshold=0.5,
                 status_penalty_exponent=2,

                 max_mates_per_female=100,
                 low_status_mating_prob=0.15,
                 epigenetic_boost_prob=0.6,
                 max_age=60,

                 peak_fertility_age_min=10,
                 peak_fertility_age_max=25,
                 low_fertility_age_min=26,
                 low_fertility_age_max=33,
                 very_low_fertility_age_min=34,
                 very_low_fertility_age_max=55,
                 infertility_threshold=55,
                 high_fertility_value=1.75,
                 medium_fertility_value=1.0,
                 low_fertility_value=0.1,
                 male_fertility_value=0.4,
                 dance_attraction_age_boost=2.0,
                 male_attraction_attempts_base=1,
                 male_attraction_attempts_boost=5,
                 max_female_mates_young=20,
                 tournament_size=15,
                 top_male_selection_prob=0.6,
                 epigenetic_transfer_prob=0.5,
                 epigenetic_self_boost=0.2,
                 behavioral_bias_prob=0.3,
                 repair_penalty_factor=100,
                 mother_inheritance_prob=0.7,
                 epigenetic_inheritance_boost=0.2,
                 epigenetic_inheritance_reduction=0.1,
                 high_fertility_offspring_min=10,
                 high_fertility_offspring_max=20,
                 medium_fertility_offspring_min=1,
                 medium_fertility_offspring_max=5,
                 chimeric_transfer_count=3):
        # parâmetros auxiliares
        self.status_decay_rate = status_decay_rate
        self.low_status_threshold = low_status_threshold
        self.status_penalty_exponent = status_penalty_exponent
        self.epigenetic_boost_prob = epigenetic_boost_prob
        self.max_age = max_age

        # fertilidade
        self.peak_fertility_age_min = peak_fertility_age_min
        self.peak_fertility_age_max = peak_fertility_age_max
        self.low_fertility_age_min = low_fertility_age_min
        self.low_fertility_age_max = low_fertility_age_max
        self.very_low_fertility_age_min = very_low_fertility_age_min
        self.very_low_fertility_age_max = very_low_fertility_age_max
        self.infertility_threshold = infertility_threshold
        self.high_fertility_value = high_fertility_value
        self.medium_fertility_value = medium_fertility_value
        self.low_fertility_value = low_fertility_value
        self.male_fertility_value = male_fertility_value

        # atração e seleção
        self.dance_attraction_age_boost = dance_attraction_age_boost
        self.male_attraction_attempts_base = male_attraction_attempts_base
        self.male_attraction_attempts_boost = male_attraction_attempts_boost
        self.max_female_mates_young = max_female_mates_young
        self.tournament_size = tournament_size
        self.top_male_selection_prob = top_male_selection_prob
        self.low_status_mating_prob = low_status_mating_prob

        # epigenética
        self.epigenetic_transfer_prob = epigenetic_transfer_prob
        self.epigenetic_self_boost = epigenetic_self_boost
        self.behavioral_bias_prob = behavioral_bias_prob
        self.mother_inheritance_prob = mother_inheritance_prob
        self.epigenetic_inheritance_boost = epigenetic_inheritance_boost
        self.epigenetic_inheritance_reduction = epigenetic_inheritance_reduction

        # reparo / penalidade
        self.repair_penalty_factor = repair_penalty_factor

        # reprodução
        self.high_fertility_offspring_min = high_fertility_offspring_min
        self.high_fertility_offspring_max = high_fertility_offspring_max
        self.medium_fertility_offspring_min = medium_fertility_offspring_min
        self.medium_fertility_offspring_max = medium_fertility_offspring_max
        self.chimeric_transfer_count = chimeric_transfer_count

        # problema
        self.instance = instance
        self.n = instance['n']
        self.W = instance['W']
        self.w = np.array(instance['w'])
        self.t = np.array(instance['t'])
        self.neighbors = self.build_neighbors()

        # biologia adicional
        self.max_mates_per_female = max_mates_per_female

        # parâmetros algorítmicos
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.max_iter = max_iter

        # inicialização
        self.population = self.initialize_population()
        num_females = self.pop_size // 2
        num_males = self.pop_size - num_females
        self.gender = np.array(['F'] * num_females + ['M'] * num_males)
        np.random.shuffle(self.gender)
        self.age = np.random.randint(0, max_age // 2, self.pop_size)
        self.fertility = np.zeros(self.pop_size)
        self.status = np.zeros(self.pop_size)
        self.dance_style = np.random.rand(self.pop_size, 5)
        self.chimeric_dna = [set() for _ in range(self.pop_size)]
        self.epigenetic_marks = [np.zeros(self.n) for _ in range(self.pop_size)]

        # históricos
        self.avg_fitness_history = []
        self.male_avg_fitness_history = []
        self.current_fitness = None
        self.mate_counts = []
        self.age_groups = {'fertile': 0, 'low_fertility': 0, 'infertile': 0}

        self.best_solution = None
        self.best_fitness = -float('inf')
        self.best_fitness_history = []

        self.initialize_attributes()
        self.update_fertility()

    # ---------------------------------------------------------------------
    # Métodos auxiliares
    # ---------------------------------------------------------------------
    def build_neighbors(self):
        """Constrói lista de vizinhos (ingredientes incompatíveis)."""
        neighbors = [[] for _ in range(self.n)]
        for j, k in self.instance['incompatible_pairs']:
            neighbors[j].append(k)
            neighbors[k].append(j)
        return neighbors

    def initialize_population(self):
        """Gera população inicial de soluções viáveis (greedy random)."""
        population = []
        ingredient_selection_prob = 0.7
        while len(population) < self.pop_size:
            sol = np.zeros(self.n, dtype=int)
            weight = 0
            indices = list(range(self.n))
            random.shuffle(indices)
            for i in indices:
                if weight + self.w[i] > self.W:
                    continue
                conflict = any(sol[j] == 1 for j in self.neighbors[i])
                if not conflict and random.random() < ingredient_selection_prob:
                    sol[i] = 1
                    weight += self.w[i]
            population.append(sol)
        return np.array(population)

    def update_fertility(self):
        """Atualiza fertilidade de acordo com idade e sexo."""
        self.age_groups = {'fertile': 0, 'low_fertility': 0, 'infertile': 0}
        for i in range(len(self.population)):
            if self.gender[i] == 'F':
                age = self.age[i]
                if self.peak_fertility_age_min <= age <= self.peak_fertility_age_max:
                    self.fertility[i] = self.high_fertility_value
                    self.age_groups['fertile'] += 1
                elif self.low_fertility_age_min <= age <= self.low_fertility_age_max:
                    self.fertility[i] = self.medium_fertility_value
                    self.age_groups['fertile'] += 1
                elif self.very_low_fertility_age_min <= age <= self.very_low_fertility_age_max:
                    self.fertility[i] = self.low_fertility_value
                    self.age_groups['low_fertility'] += 1
                elif age > self.infertility_threshold:
                    self.fertility[i] = 0.0
                    self.age_groups['infertile'] += 1
                else:
                    self.fertility[i] = 0.0
            else:
                self.fertility[i] = self.male_fertility_value

    def initialize_attributes(self, male_avg_fitness=None):
        """Inicializa status (fitness normalizado) com foco na média masculina."""
        fitness = self.evaluate_fitness()
        if male_avg_fitness is None:
            min_fit, max_fit = np.min(fitness), np.max(fitness)
            range_fit = max_fit - min_fit if max_fit - min_fit > 0 else 1
            normalized = (fitness - min_fit) / range_fit
        else:
            min_val = max(0, male_avg_fitness - male_avg_fitness / 2)
            max_val = male_avg_fitness + male_avg_fitness / 2
            range_val = max_val - min_val if max_val - min_val > 0 else 1
            normalized = np.clip((fitness - min_val) / range_val, 0, 1)
        for i in range(len(self.population)):
            if self.gender[i] == 'F':
                self.status[i] = 0.3 * normalized[i] + 0.7 * self.fertility[i]
            else:
                if male_avg_fitness:
                    deviation = abs(fitness[i] - male_avg_fitness)
                    self.status[i] = 1 / (1 + deviation / male_avg_fitness) if deviation else 1.0
                else:
                    self.status[i] = 1.0
        # estilo de dança (agregado por blocos)
        dance_groups = 5
        for i in range(len(self.population)):
            sol = self.population[i]
            group_size = max(1, self.n // dance_groups)
            for g in range(dance_groups):
                start = g * group_size
                end = min((g + 1) * group_size, self.n)
                self.dance_style[i, g] = np.mean(sol[start:end])

    def evaluate_fitness(self):
        """Função objetivo – sabor menos penalidades de peso e incompatibilidade."""
        epigenetic_boost_factor = 0.2
        fitness = []
        for i, sol in enumerate(self.population):
            flavor = np.sum(self.t * sol)
            weight = np.sum(self.w * sol)
            penalty = 0
            if weight > self.W:
                penalty = (weight - self.W) * self.repair_penalty_factor
            for j in range(self.n):
                if sol[j] == 1:
                    for k in self.neighbors[j]:
                        if sol[k] == 1:
                            penalty += self.repair_penalty_factor
                            break
            if i < len(self.epigenetic_marks):
                boosted = np.sum(self.t * sol * (1 + epigenetic_boost_factor * self.epigenetic_marks[i]))
                flavor = max(flavor, boosted)
            fitness.append(flavor - penalty)
        self.current_fitness = np.array(fitness)
        return self.current_fitness

    # ---------------------------------------------------------------------
    # Mecânicas de acasalamento e microquimerismo
    # ---------------------------------------------------------------------
    def female_dance_attraction(self, female_id, male_ids):
        """Calcula atração da fêmea para cada macho (baseado no estilo de dança)."""
        female_dance = self.dance_style[female_id]
        age_factor = 1.0
        if self.peak_fertility_age_min <= self.age[female_id] <= self.peak_fertility_age_max:
            age_factor = self.dance_attraction_age_boost
        base_randomness = 0.8
        rand_range = 0.4
        reactions = []
        for mid in male_ids:
            male_status = self.status[mid]
            male_dance = self.dance_style[mid]
            norm_f = np.linalg.norm(female_dance)
            norm_m = np.linalg.norm(male_dance)
            if norm_f > 1e-10 and norm_m > 1e-10:
                sim = np.dot(female_dance, male_dance) / (norm_f * norm_m)
            else:
                sim = 0.0
            attraction = male_status * sim * age_factor
            randomness = base_randomness + rand_range * random.random()
            reactions.append(attraction * randomness)
        return np.array(reactions)

    def mate_selection(self):
        """Seleção hierárquica onde machos competem por fêmeas férteis."""
        female_ids = np.where(self.gender == 'F')[0]
        male_ids = np.where(self.gender == 'M')[0]
        if len(female_ids) == 0 or len(male_ids) == 0:
            return []
        attraction = np.zeros((len(female_ids), len(male_ids)))
        for i, fid in enumerate(female_ids):
            attraction[i] = self.female_dance_attraction(fid, male_ids)
        suitors = defaultdict(list)
        pairs = []
        female_mate_counts = defaultdict(int)
        for local_mid, mid in enumerate(male_ids):
            attempts = self.male_attraction_attempts_base
            if self.peak_fertility_age_min <= self.age[mid] <= self.peak_fertility_age_max:
                attempts = self.male_attraction_attempts_boost
            for _ in range(attempts):
                probs = attraction[:, local_mid]
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                    fid = np.random.choice(female_ids, p=probs)
                    suitors[fid].append(mid)
        for fid in female_ids:
            if not suitors[fid] or self.fertility[fid] <= 0:
                continue
            suitors[fid].sort(key=lambda m: self.status[m], reverse=True)
            max_mates = self.max_mates_per_female
            if self.peak_fertility_age_min <= self.age[fid] <= self.peak_fertility_age_max:
                max_mates = min(self.max_female_mates_young, self.max_mates_per_female)
            selected = min(len(suitors[fid]), max_mates)
            for _ in range(selected):
                if len(suitors[fid]) > 1:
                    tsize = min(self.tournament_size, len(suitors[fid]))
                    pool = suitors[fid][:tsize]
                    chosen = pool[0] if random.random() < self.top_male_selection_prob else pool[1] if tsize > 1 else pool[0]
                else:
                    chosen = suitors[fid][0]
                pairs.append((fid, chosen))
                female_mate_counts[fid] += 1
                suitors[fid].remove(chosen)
        for fid in female_ids:
            if (random.random() < self.low_status_mating_prob and
                female_mate_counts[fid] < self.max_mates_per_female and
                self.fertility[fid] > 0):
                current = [m for f, m in pairs if f == fid]
                low_status = [mid for mid in male_ids if mid not in current and self.status[mid] < self.low_status_threshold]
                if low_status:
                    min_status = min(self.status[mid] for mid in low_status)
                    probs = [(self.status[mid] - min_status + 0.1) ** 2 for mid in low_status]
                    total = sum(probs)
                    probs = [p / total for p in probs] if total else None
                    chosen = np.random.choice(low_status, p=probs) if probs else random.choice(low_status)
                    pairs.append((fid, chosen))
                    female_mate_counts[fid] += 1
        return pairs

    def microchimerism(self, recipient_id, donor_id):
        """Transferência de DNA entre indivíduos (alpha * n genes)."""
        if recipient_id >= self.pop_size or donor_id >= self.pop_size:
            return set()
        transfer = max(1, int(self.alpha * self.n))
        gene_idx = random.sample(range(self.n), transfer)
        for idx in gene_idx:
            self.population[recipient_id][idx] = self.population[donor_id][idx]
            if random.random() < self.epigenetic_transfer_prob:
                self.epigenetic_marks[recipient_id][idx] = self.epigenetic_marks[donor_id][idx]
        self.chimeric_dna[recipient_id].update(gene_idx)
        return set(gene_idx)

    def epigenetic_self_molding(self, male_id, transferred_genes):
        """Reforço epigenético nos machos após transferência de DNA."""
        for gene in transferred_genes:
            if random.random() < self.epigenetic_boost_prob:
                self.epigenetic_marks[male_id][gene] = min(1.0, self.epigenetic_marks[male_id][gene] + self.epigenetic_self_boost)
                if random.random() < self.behavioral_bias_prob:
                    self.population[male_id][gene] = 1

    def repair_solution(self, sol):
        """Repara solução inviável usando heurística gulosa."""
        weight = np.sum(self.w * sol)
        selected = np.where(sol == 1)[0]
        while weight > self.W and selected.size > 0:
            ratios = np.where(self.w[selected] > 0, self.t[selected] / self.w[selected], np.inf)
            idx = selected[np.argmin(ratios)]
            sol[idx] = 0
            weight -= self.w[idx]
            selected = np.setdiff1d(selected, [idx])
        for i in range(self.n):
            if sol[i] == 1:
                for j in self.neighbors[i]:
                    if sol[j] == 1:
                        if self.t[i] < self.t[j]:
                            sol[i] = 0
                        else:
                            sol[j] = 0
                        break
        order = sorted(range(self.n), key=lambda i: self.t[i] / self.w[i] if self.w[i] > 0 else 0, reverse=True)
        for i in order:
            if sol[i] == 0 and weight + self.w[i] <= self.W:
                conflict = any(sol[j] == 1 for j in self.neighbors[i])
                if not conflict:
                    sol[i] = 1
                    weight += self.w[i]
        return sol

    def epigenetic_inheritance(self, mother, father, child_dna):
        """Herança epigenética influenciada pelos pais."""
        if random.random() < self.mother_inheritance_prob:
            child_epigen = self.epigenetic_marks[mother].copy()
        else:
            child_epigen = self.epigenetic_marks[father].copy()
        mother_age = self.age[mother]
        mutation_adj = 1.0
        if mother_age > self.low_fertility_age_max:
            mutation_adj = 1.0 + (mother_age - self.low_fertility_age_max) * 0.05
        for i in range(self.n):
            if random.random() < self.mutation_rate * mutation_adj:
                child_dna[i] = 1 - child_dna[i]
            if child_dna[i] == 1 and random.random() < 0.3:
                child_epigen[i] = min(1.0, child_epigen[i] + self.epigenetic_inheritance_boost)
            elif child_dna[i] == 0 and random.random() < 0.2:
                child_epigen[i] = max(0.0, child_epigen[i] - self.epigenetic_inheritance_reduction)
        return child_epigen

    def reproduce(self, pairs):
        new_pop, new_chimeric, new_epigen, new_gender, new_age = [], [], [], [], []
        for mother, father in pairs:
            if self.fertility[mother] > 0.8:
                n_off = random.randint(self.high_fertility_offspring_min, self.high_fertility_offspring_max)
            elif self.fertility[mother] > 0.2:
                n_off = random.randint(self.medium_fertility_offspring_min, self.medium_fertility_offspring_max)
            else:
                n_off = 0
            for _ in range(n_off):
                gender = 'F' if random.random() < 0.5 else 'M'
                new_gender.append(gender)
                child = np.zeros(self.n, dtype=int)
                for i in range(self.n):
                    child[i] = self.population[mother][i] if random.random() < 0.6 else self.population[father][i]
                if self.chimeric_dna[mother]:
                    genes = random.sample(list(self.chimeric_dna[mother]), min(self.chimeric_transfer_count, len(self.chimeric_dna[mother])))
                    for g in genes:
                        child[g] = self.population[mother][g]
                child_epigen = self.epigenetic_inheritance(mother, father, child)
                child = self.repair_solution(child)
                new_pop.append(child)
                new_chimeric.append(set())
                new_epigen.append(child_epigen)
                new_age.append(0)
        survivors = [i for i in range(len(self.population)) if self.age[i] < self.max_age]
        for i in survivors:
            new_pop.append(self.population[i])
            new_chimeric.append(self.chimeric_dna[i])
            new_epigen.append(self.epigenetic_marks[i])
            new_gender.append(self.gender[i])
            new_age.append(self.age[i])
        if len(new_pop) > self.pop_size:
            fitness_vals = []
            for sol in new_pop:
                flavor = np.sum(self.t * sol)
                weight = np.sum(self.w * sol)
                pen = 0
                if weight > self.W:
                    pen = (weight - self.W) * self.repair_penalty_factor
                for i in range(self.n):
                    if sol[i] == 1:
                        for j in self.neighbors[i]:
                            if sol[j] == 1:
                                pen += self.repair_penalty_factor
                                break
                fitness_vals.append(flavor - pen)
            idx = np.argsort(fitness_vals)[-self.pop_size:]
            new_pop = [new_pop[i] for i in idx]
            new_chimeric = [new_chimeric[i] for i in idx]
            new_epigen = [new_epigen[i] for i in idx]
            new_gender = [new_gender[i] for i in idx]
            new_age = [new_age[i] for i in idx]
        self.population = np.array(new_pop)
        self.chimeric_dna = new_chimeric
        self.epigenetic_marks = new_epigen
        self.gender = np.array(new_gender)
        self.age = np.array(new_age)
        self.fertility = np.zeros(len(self.population))
        self.status = np.zeros(len(self.population))
        self.update_fertility()

    # ---------------------------------------------------------------------
    # Execução principal
    # ---------------------------------------------------------------------
    def evolve(self):
        start = time.time()
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.best_fitness_history = []
        for gen in range(self.max_iter):
            self.age += 1
            self.update_fertility()
            fitness = self.evaluate_fitness()
            avg = np.mean(fitness)
            max_fit = np.max(fitness)
            male_idx = np.where(self.gender == 'M')[0]
            male_avg = np.mean(fitness[male_idx]) if male_idx.size else avg
            self.male_avg_fitness_history.append(male_avg)
            if max_fit > self.best_fitness:
                self.best_fitness = max_fit
                self.best_solution = self.population[np.argmax(fitness)].copy()
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(avg)
            self.initialize_attributes(male_avg)
            pairs = self.mate_selection()
            for mother, father in pairs:
                transferred = self.microchimerism(mother, father)
                self.epigenetic_self_molding(father, transferred)
            self.reproduce(pairs)
            if gen % 10 == 0:
                print(f"Gen {gen}: Male Avg {male_avg:.2f} | Best {self.best_fitness:.2f}")
        elapsed = time.time() - start
        flavor = np.sum(self.t * self.best_solution)
        weight = np.sum(self.w * self.best_solution)
        print(f"\nBest solution – flavor {flavor}, weight {weight}/{self.W}, fitness {self.best_fitness:.2f}, time {elapsed:.2f}s")
        return self.best_solution, flavor, self.age_groups, self.male_avg_fitness_history

    def plot_results(self, fertility_history, competition_stats, male_avg_fitness_history):
        plt.figure(figsize=(15, 15))
        plt.subplot(3, 2, 1)
        plt.plot(self.avg_fitness_history, label='Pop Avg')
        plt.plot(male_avg_fitness_history, label='Male Avg')
        plt.plot(self.best_fitness_history, label='Best')
        plt.legend(); plt.grid(alpha=0.3)
        plt.subplot(3, 2, 2)
        if fertility_history:
            gens = range(len(fertility_history))
            plt.stackplot(gens,
                         [h['fertile'] for h in fertility_history],
                         [h['low_fertility'] for h in fertility_history],
                         [h['infertile'] for h in fertility_history],
                         labels=['Fertile','Low','Infertile'])
            plt.legend(loc='upper right')
        plt.subplot(3, 2, 3)
        if self.gender.size:
            plt.boxplot([self.status[self.gender=='F'], self.status[self.gender=='M']], labels=['F','M'])
        plt.subplot(3, 2, 4)
        if self.mate_counts:
            plt.hist(self.mate_counts, bins=np.arange(0, self.max_mates_per_female+2)-0.5)
        plt.subplot(3, 2, 5)
        if competition_stats:
            plt.plot(competition_stats)
        plt.subplot(3, 2, 6)
        plt.plot(male_avg_fitness_history)
        plt.tight_layout()
        plt.savefig('evolutionary_dance_ep_results.png')
        plt.show()

# ---------------------------------------------------------------------
# Funções auxiliares de I/O
# ---------------------------------------------------------------------

def read_ep_instance(file_path):
    """Lê instância .dat usada pelos experimentos."""
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    blocks = []
    cur = []
    for line in lines:
        if line:
            cur.append(line)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    n, _, W = map(int, blocks[0][0].split())
    t = []
    for l in blocks[1]:
        t.extend(map(int, l.split()))
    w = []
    for l in blocks[2]:
        w.extend(map(int, l.split()))
    incompat = []
    if len(blocks) > 3:
        for l in blocks[3]:
            j, k = map(int, l.split())
            incompat.append((j-1, k-1))
    return {'n': n, 'W': W, 't': t, 'w': w, 'incompatible_pairs': incompat}


def run_evolutionary_dance_experiment(instances_dir, output_dir, runs=10):
    """Executa o algoritmo em todas as instâncias *.dat.

    Args:
        instances_dir: diretório contendo arquivos .dat.
        output_dir: onde salvar resultados e soluções.
        runs: número de execuções independentes por instância.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    instance_files = sorted([f for f in os.listdir(instances_dir) if f.endswith('.dat')])
    bkv = {'ep01': 2118, 'ep02': 1378, 'ep03': 2850, 'ep04': 2730, 'ep05': 2624,
           'ep06': 4690, 'ep07': 4440, 'ep08': 5020, 'ep09': 4568, 'ep10': 4390}
    results = []
    for file in instance_files:
        name = os.path.splitext(file)[0]
        instance = read_ep_instance(os.path.join(instances_dir, file))
        flavors, times, best_sols, mates, fert_hist, comp_hist, divers = [], [], [], [], [], [], []
        for _ in range(runs):
            ga = EvolutionaryDanceEP(instance,
                                     pop_size=100,
                                     mutation_rate=0.0,
                                     alpha=0.002,
                                     max_iter=150,
                                     max_mates_per_female=20,
                                     low_status_mating_prob=0.2,
                                     epigenetic_boost_prob=0.7,
                                     max_age=60,
                                     status_decay_rate=0.7,
                                     low_status_threshold=0.5,
                                     status_penalty_exponent=30,
                                     peak_fertility_age_min=10,
                                     peak_fertility_age_max=25,
                                     low_fertility_age_min=26,
                                     low_fertility_age_max=33,
                                     very_low_fertility_age_min=34,
                                     very_low_fertility_age_max=60,
                                     infertility_threshold=55,
                                     high_fertility_value=2,
                                     medium_fertility_value=1.0,
                                     low_fertility_value=0.5,
                                     male_fertility_value=0.8,
                                     dance_attraction_age_boost=5.0,
                                     male_attraction_attempts_base=1,
                                     male_attraction_attempts_boost=10,
                                     max_female_mates_young=250,
                                     tournament_size=7,
                                     top_male_selection_prob=0.8,
                                     epigenetic_transfer_prob=0.6,
                                     epigenetic_self_boost=0.3,
                                     behavioral_bias_prob=0.6,
                                     repair_penalty_factor=110,
                                     mother_inheritance_prob=0.6,
                                     epigenetic_inheritance_boost=0.6,
                                     epigenetic_inheritance_reduction=0.2,
                                     high_fertility_offspring_min=10,
                                     high_fertility_offspring_max=25,
                                     medium_fertility_offspring_min=0,
                                     medium_fertility_offspring_max=10,
                                     chimeric_transfer_count=10)
            best, flavor, fert_hist_local, comp_hist_local, male_hist = ga.evolve()
            flavors.append(flavor)
            times.append(ga.best_fitness)  # placeholder for time
            best_sols.append(best)
            mates.append(np.mean(ga.mate_counts) if ga.mate_counts else 0)
            fert_hist.append(fert_hist_local)
            comp_hist.append(np.sum(comp_hist_local))
            divers.append(np.mean([np.std(np.array(ga.population)[:, i]) for i in range(ga.n)]))
        best_flavor = max(flavors)
        idx = flavors.index(best_flavor)
        results.append({
            'instance': name,
            'best_flavor': best_flavor,
            'avg_flavor': np.mean(flavors),
            'std_flavor': np.std(flavors),
            'avg_time': np.mean(times),
            'avg_mates': np.mean(mates),
            'avg_competition': np.mean(comp_hist),
            'avg_diversity': np.mean(divers),
            'std_diversity': np.std(divers),
            'dev_bkv': 100 * (bkv.get(name, 0) - best_flavor) / bkv.get(name, 1),
            'solution': best_sols[idx]
        })
        out_path = os.path.join(output_dir, f"{name}_solution.txt")
        with open(out_path, 'w') as f:
            f.write(f"Instância: {name}\nSabor: {best_flavor}\nPeso: {np.sum(instance['w'] * best_sols[idx])}/{instance['W']}\n")
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('Instância,Melhor Sabor,Sabor Médio,Desvio,Tempo Médio,Mates Médios,Comp Média,Div Média,Div Desvio,Desvio BKV(%)\n')
        for r in results:
            f.write(f"{r['instance']},{r['best_flavor']},{r['avg_flavor']:.2f},{r['std_flavor']:.2f},{r['avg_time']:.2f},{r['avg_mates']:.2f},{r['avg_competition']:.1f},{r['avg_diversity']:.4f},{r['std_diversity']:.4f},{r['dev_bkv']:.2f}\n")
    return results
