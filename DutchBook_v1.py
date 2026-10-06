# Potreban je NumPy ≥ 2.0
# scipy



import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np


N = 39
K = 7
BASE = 40
TOTAL = math.comb(N, K)
BATCH = 50_000

# Fiksna regularizacija zajedničke distribucije:
# 50% empirijski model + 50% uniformna distribucija.
ALPHA = 0.5


def load_csv(path):
    draws = []

    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for line, row in enumerate(csv.reader(file), 1):
            if not row or all(not value.strip() for value in row):
                continue

            try:
                draw = sorted(int(value.strip()) for value in row)
            except ValueError as exc:
                raise ValueError(
                    f"Red {line}: neispravan broj."
                ) from exc

            if (
                len(draw) != K
                or len(set(draw)) != K
                or not all(1 <= value <= N for value in draw)
            ):
                raise ValueError(
                    f"Red {line}: potrebno je 7 različitih brojeva 1..39."
                )

            draws.append(draw)

    if len(draws) < 50:
        raise ValueError("Potrebno je najmanje 50 izvlačenja.")

    return np.asarray(draws, dtype=np.int64)


def build_table(draws, order):
    """
    Tačan prikaz empirijske distribucije zajedničkih podskupova.

    Svako istorijsko izvlačenje ima jednaku težinu.
    Ne rangiraju se pojedinačni brojevi.
    """
    weights = np.zeros(BASE ** order, dtype=np.int64)

    for positions in itertools.combinations(range(K), order):
        keys = np.zeros(len(draws), dtype=np.int64)

        for position in positions:
            keys = keys * BASE + draws[:, position]

        np.add.at(weights, keys, 1)

    assert int(weights.sum()) == len(draws) * math.comb(K, order)

    return weights


def joint_score(combinations, weights, order):
    """
    Za kombinaciju S računa:

        sum_D C(|S presek D|, order)

    D prolazi kroz sva istorijska izvlačenja.

    Tabela omogućava isti tačan račun bez ponovnog prolaska
    kroz ceo CSV za svaku kandidat-kombinaciju.
    """
    scores = np.zeros(len(combinations), dtype=np.int64)

    for positions in itertools.combinations(range(K), order):
        keys = np.zeros(len(combinations), dtype=np.int64)

        for position in positions:
            keys = keys * BASE + combinations[:, position]

        scores += weights[keys]

    return scores


def normalizer(rows, order):
    """
    Tačna normalizacija empirijske zajedničke distribucije.

    Za svako D:
      sum_S C(|S presek D|, order)
      = C(7, order) * C(39 - order, 7 - order).

    Zato zbir P(S) preko svih validnih sedmorki iznosi 1.
    Sve marginalne i uslovne verovatnoće izvedene iz iste
    raspodele zadovoljavaju Dutch Book koherentnost.
    """
    return (
        rows
        * math.comb(K, order)
        * math.comb(N - order, K - order)
    )


def probabilities(combinations, weights, order, rows):
    empirical = (
        joint_score(combinations, weights, order)
        / normalizer(rows, order)
    )

    return ALPHA * empirical + (1.0 - ALPHA) / TOTAL


def choose_order(draws):
    """
    Hronološka provera:
    prvih 80% redova za obuku, poslednjih 20% za proveru.

    Izbor prema prosečnoj log-verovatnoći stvarno izvučenih
    sedmorki, bez korišćenja tih redova tokom obuke.
    """
    split = int(0.8 * len(draws))
    training = draws[:split]
    validation = draws[split:]

    selected = None
    best_log_score = -math.inf

    for order in (2, 3, 4):
        weights = build_table(training, order)

        prediction = probabilities(
            validation,
            weights,
            order,
            len(training),
        )

        log_score = float(np.log(prediction).mean())

        print(
            f"Red zavisnosti {order}: "
            f"provera log P = {log_score:.9f}",
            flush=True,
        )

        # Kod jednakih rezultata prednost ima manji red.
        if log_score > best_log_score:
            selected = order
            best_log_score = log_score

    uniform_score = -math.log(TOTAL)

    print(
        f"Uniformna referenca: {uniform_score:.9f}",
        flush=True,
    )

    if best_log_score <= uniform_score:
        print(
            "Provera nije pokazala prednost "
            "nad uniformnom distribucijom.",
            flush=True,
        )

    return selected


def find_next(draws, order):
    """
    Pregled svih 15.380.937 kombinacija.

    Globalni maksimum, bez slučajnih kandidata.
    Kod jednakih maksimuma bira leksikografski prvu sedmorku.
    """
    weights = build_table(draws, order)

    iterator = itertools.combinations(range(1, N + 1), K)

    best = None
    best_value = -1
    examined = 0
    ties = 0

    while True:
        batch = list(itertools.islice(iterator, BATCH))

        if not batch:
            break

        combinations = np.asarray(batch, dtype=np.int64)
        scores = joint_score(combinations, weights, order)

        index = int(np.argmax(scores))
        value = int(scores[index])

        if value > best_value:
            best_value = value
            best = batch[index]
            ties = int(np.count_nonzero(scores == value))

        elif value == best_value:
            ties += int(np.count_nonzero(scores == value))

        examined += len(batch)

    assert examined == TOTAL
    assert len(best) == K
    assert len(set(best)) == K

    probability = (
        ALPHA * best_value / normalizer(len(draws), order)
        + (1.0 - ALPHA) / TOTAL
    )

    return best, probability, ties


def main():
    parser = argparse.ArgumentParser(
        description="Dutch Book Loto 7/39 v1"
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default="/Users/4c/Desktop/GHQ/data/loto7_4696_k79.csv",
        # default="/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_2970.csv",
        # default="/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_plus_1726.csv",
    )

    parser.add_argument(
        "--newest-first",
        action="store_true",
        help="Koristi ako prvi red sadrži najnovije izvlačenje.",
    )

    args = parser.parse_args()

    draws = load_csv(args.csv)

    # Podrazumevani redosled: najstarije -> najnovije.
    if args.newest_first:
        draws = draws[::-1].copy()

    order = choose_order(draws)

    print(
        f"Red zavisnosti: {order}; "
        f"svih {len(draws)} izvlačenja; "
        f"pregled {TOTAL} kombinacija.",
        flush=True,
    )

    prediction, probability, ties = find_next(draws, order)

    print("\nNEXT:", " ".join(map(str, prediction)))
    print(f"Verovatnoća po modelu: {probability:.12g}")
    print(f"Broj jednakih maksimuma: {ties}")


if __name__ == "__main__":
    main()



"""
Red zavisnosti 2: provera log P = -16.548069951
Red zavisnosti 3: provera log P = -16.549218477
Red zavisnosti 4: provera log P = -16.554044697
Uniformna referenca: -16.548639443
Red zavisnosti: 2; svih 4696 izvlačenja; pregled 15380937 kombinacija.

NEXT: 7 8 11 23 26 33 34
Verovatnoća po modelu: 7.00545152164e-08
Broj jednakih maksimuma: 1





Red zavisnosti 2: provera log P = -16.548109424
Red zavisnosti 3: provera log P = -16.549167224
Red zavisnosti 4: provera log P = -16.556434796
Uniformna referenca: -16.548639443
Red zavisnosti: 2; svih 2970 izvlačenja; pregled 15380937 kombinacija.

NEXT: 5 8 16 22 23 28 33
Verovatnoća po modelu: 7.02649332543e-08
Broj jednakih maksimuma: 1





Red zavisnosti 2: provera log P = -16.547665192
Red zavisnosti 3: provera log P = -16.548070916
Red zavisnosti 4: provera log P = -16.560991897
Uniformna referenca: -16.548639443
Red zavisnosti: 2; svih 1726 izvlačenja; pregled 15380937 kombinacija.

NEXT: 2 7 8 11 23 29 37
Verovatnoća po modelu: 7.17178177904e-08
Broj jednakih maksimuma: 1
"""



"""
Dutch Book kao ograničenje modela koji iz istorije računa verovatnoće, pa iz tog modela bira sledeću kombinaciju.
Postupak bi bio:
1. Iz prethodnih izvlačenja računam početne procene verovatnoća pojedinačnih brojeva i parova brojeva.
2. Dutch Book korak: korigujem te procene tako da sve mogu poticati iz jedne zajedničke raspodele nad kombinacijama od tačno sedam brojeva. Tako pojedinačne procene i procene parova ne mogu međusobno protivrečiti.
3. Među raspodelama koje zadovoljavaju te uslove biram onu sa maksimalnom entropijom — time izbegavam dodavanje obrazaca koje podaci ne podržavaju.
4. Kao predikciju vraćam kombinaciju sa najvećom verovatnoćom u toj raspodeli.
Istorija daje signal, Dutch Book obezbeđuje doslednost, maksimalna entropija određuje raspodelu, a njena najverovatnija sedmorka postaje predikcija. 


ceo CSV, bez slučajnog uzorkovanja i bez rangiranja „najčešćih” brojeva.
Model treba da uči zajedničku distribuciju sedmočlanih kombinacija, uključujući zavisnosti među brojevima. Dutch Book uslovi obezbeđuju da sve izvedene verovatnoće budu dosledne toj distribuciji. Rezultat je deterministički: isti CSV daje istu predikciju, uz unapred definisano pravilo ako više kombinacija ima isti maksimum.

Kod je izvršen na mom CSV-u i pronašao je jedan jedinstveni maksimum.
Model koristi empirijsku zajedničku distribuciju: svako istorijsko izvlačenje doprinosi kombinacijama koje sa njim dele parove, trojke ili četvorke. Hronološka provera bira red zavisnosti, a završni model koristi svih 4.696 izvlačenja. Nema normalne raspodele ni random uzorkovanja.
Dutch Book doslednost sledi iz tačno normalizovane distribucije nad validnim sedmorkama. Na proveri je izabrana zavisnost parova, sa vrlo malom prednošću nad uniformnom distribucijom.
"""
