import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import random
import copy

def hamming_distance(ind1, ind2):
    """Hamming distance between two binary lists."""
    return sum(a != b for a, b in zip(ind1, ind2))


def HUX_crossover(ind1, ind2, random_state=None):
    """
    Half Uniform Crossover (HUX).
    Swaps exactly half of the differing bits.
    If the number of differing bits is odd, randomly choose floor or ceil.
    """
    rng = np.random.RandomState(random_state)
    diff_positions = [i for i in range(len(ind1)) if ind1[i] != ind2[i]]
    d = len(diff_positions)
    if d > 1:
        if d % 2 == 0:
            n_swap = d // 2
        else:
            # Randomly decide to swap floor(d/2) or ceil(d/2)
            n_swap = d // 2 + rng.choice([0, 1])
        selected = rng.choice(diff_positions, n_swap, replace=False)
        for i in selected:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def evaluate_individual(
        individual,
        X,
        y,
        knn_params,
        cv_folds=5,
        random_state=42):
    """
    Evaluate a feature subset using KNN with cross-validation.
    Returns balanced accuracy.
    """
    selected = [i for i, val in enumerate(individual) if val == 1]
    if len(selected) == 0:
        return 0.0  # No features selected -> worst fitness

    clf = KNeighborsClassifier(**knn_params)
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state)
    scores = []
    for train_idx, val_idx in skf.split(X[:, selected], y):
        X_fold_train = X[train_idx][:, selected]
        y_fold_train = y[train_idx]
        X_fold_val = X[val_idx][:, selected]
        y_fold_val = y[val_idx]
        clf.fit(X_fold_train, y_fold_train)
        y_pred = clf.predict(X_fold_val)
        scores.append(balanced_accuracy_score(y_fold_val, y_pred))
    return np.mean(scores)


def create_population(pop_size, n_features, rng):
    """Create random binary population."""
    return [rng.choice([0, 1], size=n_features).tolist()
            for _ in range(pop_size)]


def restart_population(elite, pop_size, flip_prob=0.35, rng=None):
    """
    Cataclysmic restart: keep elite, generate others by flipping bits of elite.
    """
    if rng is None:
        rng = np.random.RandomState()
    new_pop = [elite.copy()]
    while len(new_pop) < pop_size:
        mutant = elite.copy()
        for i in range(len(mutant)):
            if rng.rand() < flip_prob:
                mutant[i] = 1 - mutant[i]
        new_pop.append(mutant)
    return new_pop


def CHC_evolution(population, T, max_gens, X, y, knn_params, cv_folds=5,
                  verbose=False, rng=None):
    """
    Run CHC evolution on given data.
    Returns: best individual, best fitness, final population.
    """
    if rng is None:
        rng = np.random.RandomState()
    pop = copy.deepcopy(population)
    pop_size = len(pop)
    n_features = len(pop[0])

    # Initial fitness evaluation
    fitness = [
        evaluate_individual(
            ind,
            X,
            y,
            knn_params,
            cv_folds) for ind in pop]
    best_idx = np.argmax(fitness)
    best_ind = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    for gen in range(max_gens):
        # Generate offspring via HUX (only pairs with Hamming > T)
        offspring = []
        indices = list(range(pop_size))
        rng.shuffle(indices)
        for i in range(0, pop_size - 1, 2):
            p1 = pop[indices[i]]
            p2 = pop[indices[i + 1]]
            if hamming_distance(p1, p2) > T:
                c1, c2 = p1.copy(), p2.copy()
                HUX_crossover(c1, c2, random_state=rng.randint(0, 1e6))
                offspring.extend([c1, c2])

        if len(offspring) == 0:
            # Cataclysmic restart
            pop = restart_population(
                best_ind, pop_size, flip_prob=0.35, rng=rng)
            T = max(0, T - 1)
            # Re-evaluate fitness for the new population
            fitness = [
                evaluate_individual(
                    ind,
                    X,
                    y,
                    knn_params,
                    cv_folds) for ind in pop]
            best_idx = np.argmax(fitness)
            best_fit = fitness[best_idx]
            best_ind = pop[best_idx].copy()
            # No offspring this generation, skip the rest of the loop
            continue
        else:
            # Evaluate offspring
            off_fitness = [evaluate_individual(ind, X, y, knn_params, cv_folds)
                           for ind in offspring]
            # Environmental selection: keep best pop_size individuals
            combined = pop + offspring
            combined_fitness = fitness + off_fitness
            sorted_idx = np.argsort(combined_fitness)[::-1]
            pop = [combined[i] for i in sorted_idx[:pop_size]]
            fitness = [combined_fitness[i] for i in sorted_idx[:pop_size]]

        # Update best
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fit:
            best_fit = fitness[current_best_idx]
            best_ind = pop[current_best_idx].copy()

        if verbose and gen % 5 == 0:
            print(f"    Gen {gen:3d}: best fitness = {best_fit:.4f}, T = {T}")

    return best_ind, best_fit, pop


def SAGA(X_train, y_train, X_test, y_test, knn_params,
         pop_size_init=40, b=4, a=2, pr=0.5, step=10,
         cv_folds=5, random_state=42, verbose=False):
    """
    Surrogate-Assisted Genetic Algorithm for wrapper feature selection using KNN.
    Strictly follows the article specification.
    """
    rng = np.random.RandomState(random_state)
    N, K = X_train.shape

    # Precompute surrogate sample sizes (paper's sequence)
    sample_sizes = [N // (a ** i) for i in range(b, 0, -1)]
    pop_sizes = [int(pop_size_init * (pr ** i)) for i in range(b)]
    P_final = pop_sizes[-1]  # Final stage population size

    # Global best (evaluated on full training data)
    best_individual = None
    best_score_full = -np.inf

    # Store logs
    log = []

    # Initial elite (will be updated)
    elite = None

    # ========== Surrogate Levels ==========
    for level in range(b):
        S_level = sample_sizes[level]
        P_level = pop_sizes[level]
        T = K // 4  # Reset Hamming threshold for each level

        # Sample surrogate data
        idx = rng.choice(N, size=S_level, replace=False)
        X_surr, y_surr = X_train[idx], y_train[idx]

        if verbose:
            print(f"\n--- Level {level + 1}/{b} ---")
            print(f"  Sample size: {S_level} ({S_level / N:.1%} of full data)")
            print(f"  Population size: {P_level}")

        # Initialize population for this level
        if level == 0:
            pop = create_population(P_level, K, rng)
        else:
            # Use elite from previous level (best on full data so far)
            # Generate others with bit probability = ones_ratio in elite
            ones_ratio = sum(elite) / K
            pop = [elite.copy()]
            while len(pop) < P_level:
                new_ind = [
                    1 if rng.rand() < ones_ratio else 0 for _ in range(K)]
                pop.append(new_ind)

        # Level evolution loop
        gen_in_level = 0
        while True:
            # Run CHC for 'step' generations on surrogate data
            best_surr, _, pop = CHC_evolution(
                pop, T, step, X_surr, y_surr, knn_params,
                cv_folds=cv_folds, verbose=False, rng=rng
            )
            gen_in_level += step

            # Evaluate best individual on FULL training data
            score_full = evaluate_individual(best_surr, X_train, y_train,
                                             knn_params, cv_folds=cv_folds)
            if verbose:
                print(
                    f"  After {gen_in_level:3d} gens: full-data score = {score_full:.4f}")

            # Check for improvement
            if score_full > best_score_full:
                best_score_full = score_full
                best_individual = best_surr.copy()
                elite = best_surr.copy()
                # if verbose:
                #     print(
                #         f"  *** Improvement! New best score = {
                #             best_score_full:.4f}, " f"#features = {
                #             sum(best_individual)}")
                # Continue current level (stay)
            else:
                if verbose:
                    print(f"  No improvement. Switching to next level.")
                break  # Exit level loop and go to next level

        # End of level: update population for next level (if not last)
        # (population already updated in CHC_evolution; elite stored)

    # ========== Final Exploitation Stage (Convergence‑Based) ==========
    if verbose:
        print(f"\n=== Final Stage: Full Data Exploitation ===")
        print(f"  Population size: {P_final}")

    # Build final population from best individual
    final_pop = [best_individual.copy()]
    ones_ratio = sum(best_individual) / K
    while len(final_pop) < P_final:
        new_ind = [1 if rng.rand() < ones_ratio else 0 for _ in range(K)]
        final_pop.append(new_ind)

    # Run CHC on full data until convergence
    # Convergence criterion: stop after 3 consecutive restarts with no improvement
    # in full‑data score, or after max_gens_final generations.
    max_gens_final = 200
    T = K // 4
    pop = copy.deepcopy(final_pop)
    pop_size = P_final

    # Initial evaluation
    fitness = [evaluate_individual(ind, X_train, y_train, knn_params, cv_folds)
               for ind in pop]
    best_idx = np.argmax(fitness)
    best_ind = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    best_score_full_final = best_fit

    restart_count = 0
    no_improve_restarts = 0
    gen = 0

    while gen < max_gens_final and no_improve_restarts < 3:
        # Generate offspring
        offspring = []
        indices = list(range(pop_size))
        rng.shuffle(indices)
        for i in range(0, pop_size - 1, 2):
            p1 = pop[indices[i]]
            p2 = pop[indices[i + 1]]
            if hamming_distance(p1, p2) > T:
                c1, c2 = p1.copy(), p2.copy()
                HUX_crossover(c1, c2, random_state=rng.randint(0, 1e6))
                offspring.extend([c1, c2])

        if len(offspring) == 0:
            # Cataclysmic restart
            pop = restart_population(
                best_ind, pop_size, flip_prob=0.35, rng=rng)
            T = max(0, T - 1)
            restart_count += 1
            # Re-evaluate fitness for the new population
            fitness = [
                evaluate_individual(
                    ind,
                    X_train,
                    y_train,
                    knn_params,
                    cv_folds) for ind in pop]
            best_idx = np.argmax(fitness)
            best_fit = fitness[best_idx]
            best_ind = pop[best_idx].copy()
            # Also update full‑data best score
            score_full_final = evaluate_individual(
                best_ind, X_train, y_train, knn_params, cv_folds=cv_folds)
            if score_full_final > best_score_full_final:
                best_score_full_final = score_full_final
                best_individual = best_ind.copy()
                no_improve_restarts = 0
            else:
                no_improve_restarts += 1
            # skip the rest of this generation (offspring already handled)
            continue
        else:
            # Evaluate offspring
            off_fitness = [
                evaluate_individual(
                    ind,
                    X_train,
                    y_train,
                    knn_params,
                    cv_folds) for ind in offspring]
            combined = pop + offspring
            combined_fitness = fitness + off_fitness
            sorted_idx = np.argsort(combined_fitness)[::-1]
            pop = [combined[i] for i in sorted_idx[:pop_size]]
            fitness = [combined_fitness[i] for i in sorted_idx[:pop_size]]
            restart_count = 0  # Reset because we had offspring

        # Update best
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fit:
            best_fit = fitness[current_best_idx]
            best_ind = pop[current_best_idx].copy()
            # Improvement in surrogate fitness; check full‑data score
            score_full_final = evaluate_individual(
                best_ind, X_train, y_train, knn_params, cv_folds=cv_folds)
            if score_full_final > best_score_full_final:
                best_score_full_final = score_full_final
                best_individual = best_ind.copy()
                no_improve_restarts = 0
                # if verbose:
                #     print(
                #         f"  Final stage gen {gen}: full‑data score improved to {
                #             best_score_full_final:.4f}")

        gen += 1

        # if verbose and gen % 20 == 0:
        #     print(
        #         f"  Final stage gen {gen}: best full‑data score = {
        #             best_score_full_final:.4f}, restarts = {restart_count}")

    # Use the best found during final stage
    if best_score_full_final > best_score_full:
        best_individual = best_ind
        best_score_full = best_score_full_final

    # ========== Final Evaluation on Test Set ==========
    selected = [i for i, v in enumerate(best_individual) if v == 1]
    if len(selected) > 0:
        clf = KNeighborsClassifier(**knn_params)
        clf.fit(X_train[:, selected], y_train)
        y_pred_test = clf.predict(X_test[:, selected])
        test_acc = balanced_accuracy_score(y_test, y_pred_test)
        if verbose:
            print(f"\nFinal test balanced accuracy: {test_acc:.4f}")
            print(f"Number of selected features: {len(selected)}")
    else:
        test_acc = 0.0
        if verbose:
            print("No features selected!")

    return best_individual, selected, test_acc


def select_top_k_features(
        best_individual,
        X_train,
        y_train,
        knn_params,
        k=30,
        cv_folds=5):
    """
    Select exactly k features from best_individual based on individual feature importance.
    Uses leave-one-out evaluation: removes each feature and measures performance drop.
    """
    selected = [i for i, v in enumerate(best_individual) if v == 1]

    if len(selected) <= k:
        # If already k or fewer features, return as is
        return selected

    # Evaluate baseline with all selected features
    baseline_score = evaluate_individual(
        best_individual, X_train, y_train, knn_params, cv_folds)

    # Calculate importance for each selected feature
    feature_importance = []
    for idx in selected:
        # Create mask without this feature
        temp_mask = best_individual.copy()
        temp_mask[idx] = 0
        score_without = evaluate_individual(
            temp_mask, X_train, y_train, knn_params, cv_folds)
        importance = baseline_score - score_without  # Performance drop
        feature_importance.append((idx, importance))

    # Sort by importance (descending) and select top k
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in feature_importance[:k]]

    return sorted(top_k_indices)