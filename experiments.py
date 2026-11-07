#!/usr/bin/env python3
"""Run csma-v3, find two empirical boundaries, write tables and three figures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from scipy.stats import t as student

METRICS = ("loss", "retries", "q_mean", "q_max", "util")


def interval(rows, metric, tail):
    values = [r[metric] for r in rows]
    mean = statistics.mean(values)
    # All-zero samples are not evidence of a zero population mean.
    if mean == 0:
        return dict(mean=0.0, half=None, low=None, high=None, relative=None, ok=True)
    half = float(student.isf(tail / 2, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    return dict(mean=mean, half=half, low=max(0, mean - half), high=mean + half,
                relative=half / mean, ok=half / mean <= 0.05)


class Study:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        if args.binary:
            self.binary = args.binary.resolve()
            version = "user-supplied binary"
        else:
            root = args.ns3.resolve()
            candidates = [p for p in (root / "build" / "scratch").rglob("*csma-v3*")
                          if p.is_file() and os.access(p, os.X_OK)]
            if len(candidates) != 1:
                raise ValueError("Build scratch/csma-v3.cc first, or specify --binary explicitly")
            self.binary = candidates[0]
            version = (root / "VERSION").read_text().strip() if (root / "VERSION").exists() else "unknown"
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ValueError(f"Executable not found: {self.binary}")
        self.cache = {}
        self.summaries = {}
        self.raw = self.out / "raw.jsonl"
        # Prevent accidentally mixing builds or random streams on restart.
        config = dict(type="config", binary_sha256=hashlib.sha256(self.binary.read_bytes()).hexdigest(),
                      model_sha256=hashlib.sha256(Path(__file__).with_name("csma-v3.cc").read_bytes()).hexdigest(),
                      seed=args.seed, ns3_version=version)
        if self.raw.exists():
            with self.raw.open() as f:
                if json.loads(next(f)) != config:
                    raise ValueError("Existing raw.jsonl belongs to another build/seed; use another --out")
                for line in f:
                    row = json.loads(line)
                    key = (row["dist"], row["alpha"], row["n"], row["phase"])
                    self.cache.setdefault(key, []).append(row)
        else:
            self.raw.write_text(json.dumps(config) + "\n")
        self.version = version
        self.looks = []
        count = args.min_runs
        while count < args.max_runs:
            self.looks.append(count)
            count *= 2
        self.looks.append(args.max_runs)
        # Bonferroni over all planned looks and five means at a fixed N.
        # This avoids silently treating repeatedly inspected 99% t intervals
        # as a fixed-sample procedure. Coverage remains a CLT/t approximation.
        self.tail = 0.01 / (len(self.looks) * len(METRICS))

    def sample(self, dist, alpha, n, phase, count):
        key = (dist, alpha, n, phase)
        rows = self.cache.setdefault(key, [])
        if phase == "measure" and len(rows) > self.args.max_runs:
            raise ValueError("--max-runs is smaller than the cached sample; do not discard observations")
        def simulate(rep):
            # Separate streams for scenarios, pilot/measurement and replications.
            seed = self.args.seed + 2 * ((0 if dist == "exp" else 1) * 3 + [10, 50, 100].index(alpha))
            seed += phase == "measure"
            run = n * 1000000 + rep + 1
            command = [str(self.binary), f"--n={n}", f"--alpha={alpha}", f"--dist={dist}",
                       f"--seed={seed}", f"--run={run}"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=self.args.timeout)
            if result.returncode:
                raise RuntimeError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
            records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            if len(records) != 1:
                raise RuntimeError(f"Expected one JSON record, got: {result.stdout}")
            row = records[0]
            if (row["n"], row["alpha"], row["dist"], row["seed"], row["run"]) != (n, alpha, dist, seed, run):
                raise RuntimeError("Wrong run identity in model output")
            if (row["generated"] != row["lost"] + row["tx_end"] or
                    row["lost"] != row["queue_drops"] + row["phy_drops"] or
                    row["admitted"] != row["generated"] - row["queue_drops"] or
                    row["unfinished_at_10"] < row["queued_at_10"] or
                    not 0 <= row["util"] <= 1 or not 0 <= row["q_max"] <= 20):
                raise RuntimeError("Model balance/range check failed")
            if any(not math.isfinite(row[m]) or row[m] < 0 for m in METRICS):
                raise RuntimeError("Non-finite/negative model statistic")
            row["phase"] = phase
            return row

        if len(rows) < count:
            # Keep seed/run identities and file order independent of worker count.
            with ThreadPoolExecutor(max_workers=self.args.jobs) as pool, self.raw.open("a") as f:
                for row in pool.map(simulate, range(len(rows), count)):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    rows.append(row)
                    if len(rows) % 128 == 0:
                        print(f"  {phase}: {dist}, alpha={alpha}, N={n}, runs={len(rows)}", flush=True)
        return rows

    def measure(self, dist, alpha, n):
        key = (dist, alpha, n)
        if key in self.summaries:
            return self.summaries[key]
        existing = len(self.cache.get((*key, "measure"), []))
        for count in self.looks:
            if count < existing:
                continue
            rows = self.sample(dist, alpha, n, "measure", count)
            ci = {m: interval(rows, m, self.tail) for m in METRICS}
            r = ci["retries"]
            resolved = r["mean"] == 0 or r["high"] <= 2 or r["low"] > 2
            # The assignment requests an observed maximum, not its population mean.
            precise = all(ci[m]["ok"] for m in METRICS if m != "q_max")
            if precise and resolved:
                break
        summary = dict(dist=dist, alpha=alpha, n=n, runs=len(rows), ci=ci,
                       lost=sum(r["lost"] for r in rows),
                       generated=sum(r["generated"] for r in rows),
                       loss_runs=sum(r["lost"] > 0 for r in rows),
                       q_max_observed=max(r["q_max"] for r in rows),
                       precise=precise, retries_resolved=resolved)
        self.summaries[key] = summary
        print(f"{dist:7} alpha={alpha:3} N={n:5} R={len(rows):4} "
              f"loss={ci['loss']['mean']:.5g} retry={r['mean']:.5g} "
              f"precision={'OK' if precise else 'NOT MET'}", flush=True)
        return summary

    def predicate(self, dist, alpha, kind, phase):
        def evaluate(n):
            if phase == "pilot":
                rows = self.sample(dist, alpha, n, phase, self.args.search_runs)
                return (all(r["lost"] == 0 for r in rows) if kind == "loss" else
                        statistics.mean(r["retries"] for r in rows) <= 2)
            s = self.measure(dist, alpha, n)
            if kind == "loss":
                return s["lost"] == 0
            r = s["ci"]["retries"]
            if r["mean"] == 0 or r["high"] <= 2:
                return True
            if r["low"] > 2:
                return False
            return None  # CI straddles 2: no invented boundary decision.
        return evaluate

    def boundary(self, evaluate, start):
        # Geometric bracketing followed by integer bisection; no full scan.
        # Monotonicity is a working assumption, checked at the sampled N later.
        if evaluate(1) is not True:
            return dict(n=None, next_n=1, status="no_lower_bound")
        low = 1
        high = max(2, min(start, self.args.max_n))
        while True:
            verdict = evaluate(high)
            if verdict is None:
                return dict(n=low, next_n=high, status="uncertain")
            if not verdict:
                break
            low = high
            if high == self.args.max_n:
                return dict(n=low, next_n=None, status="max_n_reached")
            high = min(self.args.max_n, high * 2)
        while high - low > 1:
            mid = (low + high) // 2
            verdict = evaluate(mid)
            if verdict is None:
                return dict(n=low, next_n=high, status="uncertain")
            if verdict:
                low = mid
            else:
                high = mid
        return dict(n=low, next_n=high, status="adjacent_checked")

    def run_case(self, dist, alpha):
        boundaries = {}
        for kind in ("loss", "retries"):
            pilot = self.boundary(self.predicate(dist, alpha, kind, "pilot"), 4)
            guess = pilot["n"] or 2
            boundaries[kind] = self.boundary(self.predicate(dist, alpha, kind, "measure"), guess)
        # A small set of plot points, plus neighbours and upper-side checks.
        points = {1}
        for b in boundaries.values():
            if b["n"]:
                n = b["n"]
                points.update(max(1, min(self.args.max_n, round(n * factor)))
                              for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2))
                points.update(k for k in range(n - 2, n + 3) if 1 <= k <= self.args.max_n)
        for n in sorted(points):
            self.measure(dist, alpha, n)
        for kind, b in boundaries.items():
            evaluate = self.predicate(dist, alpha, kind, "measure")
            tested = sorted(n for d, a, n in self.summaries if (d, a) == (dist, alpha))
            rejected = False
            for n in tested:
                verdict = evaluate(n)
                if verdict is False:
                    rejected = True
                elif verdict is True and rejected:
                    b["status"] = "nonmonotone_observed"
            if b["status"] == "adjacent_checked":
                if not all(self.measure(dist, alpha, k)["precise"] for k in (b["n"], b["next_n"])):
                    b["status"] = "precision_not_met"
        # Report the largest tested loss-free N, including a noisy reversal.
        measured = [s for (d, a, _), s in self.summaries.items() if (d, a) == (dist, alpha)]
        loss_free = [s["n"] for s in measured if s["lost"] == 0]
        if loss_free:
            boundaries["loss"]["n"] = max(loss_free)
            upper = [s["n"] for s in measured if s["n"] > max(loss_free) and s["lost"] > 0]
            boundaries["loss"]["next_n"] = min(upper) if upper else None
        return dict(dist=dist, alpha=alpha, **boundaries)

    def write_results(self, cases):
        summaries = sorted(self.summaries.values(), key=lambda s: (s["alpha"], s["dist"], s["n"]))
        fields = ["dist", "alpha", "n", "runs", "lost", "generated", "loss_runs",
                  "q_max_observed", "precise", "retries_resolved"]
        fields += [f"{m}_{k}" for m in METRICS for k in ("mean", "low", "high", "half", "relative")]
        with (self.out / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in summaries:
                row = {k: s[k] for k in fields if k in s}
                row.update({f"{m}_{k}": s["ci"][m][k] for m in METRICS
                            for k in ("mean", "low", "high", "half", "relative")})
                writer.writerow(row)
        (self.out / "boundaries.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")

        def value(s, m):
            c = s["ci"][m]
            return "0 (не наблюдалось)" if c["mean"] == 0 else f"{c['mean']:.5g} ± {c['half']:.2g}"

        lines = ["# Реальные результаты csma-v3", "", f"Версия ns-3: {self.version}.", "",
                 "Интервалы Стьюдента с поправкой на число просмотров и пять средних; "
                 "приближённое совместное покрытие не ниже 99% для фиксированного N. "
                 "Точность проверяется по исходной полуширине, до ограничения нижней границы нулём. "
                 "Критерий ±5% относится к потерям, повторам, средней очереди и занятости; "
                 "Qmax — наблюдаемый максимум, точность среднего Qmax не требуется.", "",
                 "| α | Закон | Наибольшее N без потерь (измерительные) | Прогоны | N справа | Потери справа | Статус |",
                 "|---:|---|---:|---:|---:|---:|---|"]
        for case in cases:
            d, a, b = case["dist"], case["alpha"], case["loss"]
            s = self.summaries.get((d, a, b["n"]))
            nxt = self.summaries.get((d, a, b["next_n"]))
            lines.append(f"| {a} | {d} | {b['n']} | {s['runs'] if s else '—'} | "
                         f"{b['next_n']} | {nxt['lost'] if nxt else '—'} | {b['status']} |")
        lines += ["", "| α | Закон | N при retries ≤ 2 | retries: среднее ± h | Qmax наблюд. | Статус |",
                  "|---:|---|---:|---|---:|---|"]
        for case in cases:
            d, a, b = case["dist"], case["alpha"], case["retries"]
            s = self.summaries.get((d, a, b["n"]))
            lines.append(f"| {a} | {d} | {b['n']} | {value(s, 'retries') if s else '—'} | "
                         f"{s['q_max_observed'] if s else '—'} | {b['status']} |")
        lines += ["", "Только adjacent_checked означает проверенную соседнюю границу при принятой "
                  "монотонности. uncertain — порог или соседняя граница не подтверждены; "
                  "для повторов ДИ может пересекать 2. precision_not_met — точность "
                  "5% не достигнута; max_n_reached — верхняя граница поиска не найдена; "
                  "nonmonotone_observed — случайная немонотонность наблюдений не позволяет "
                  "объявить глобальный максимум; это не означает ошибку модели. "
                  "Для потерь N — максимум только среди исследованных измерительных точек.", "",
                  "Отсутствие потерь во всех прогонах означает только отсутствие наблюдаемых потерь. "
                  "Не является доказательством нулевой вероятности. Число повторений влияет на такую границу. "
                  "Для нулевых выборок численный ДИ среднего не оценивается.", "",
                  "| α | Закон | N | R | Потери ± h | Повторы ± h | Qср ± h | Среднее Qmax ± h | Qmax наблюд. | Занятость ± h | ±5% |",
                  "|---:|---|---:|---:|---|---|---|---|---:|---|---|"]
        for s in summaries:
            lines.append(f"| {s['alpha']} | {s['dist']} | {s['n']} | {s['runs']} | "
                         + " | ".join(value(s, m) for m in METRICS[:-1])
                         + f" | {s['q_max_observed']} | {value(s, 'util')} | {'да' if s['precise'] else 'НЕТ'} |")
        (self.out / "results.md").write_text("\n".join(lines) + "\n")
        self.plot(summaries)

    def plot(self, summaries):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for alpha in self.args.alphas:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for dist, color in (("exp", "tab:blue"), ("uniform", "tab:orange")):
                points = [s for s in summaries if s["alpha"] == alpha and s["dist"] == dist]
                for ax, metric, label in zip(axes.flat, ("loss", "retries", "q_mean", "util"),
                                             ("Доля потерь", "Повторов / принятый пакет", "Очередь, пакеты", "Доля занятости")):
                    if not points:
                        continue
                    ax.errorbar([s["n"] for s in points], [s["ci"][metric]["mean"] for s in points],
                                yerr=[[min(s["ci"][metric]["mean"], s["ci"][metric]["half"] or 0)
                                       for s in points],
                                      [s["ci"][metric]["half"] or 0 for s in points]],
                                fmt="o-", ms=3, capsize=2, label=dist, color=color)
                    if metric == "q_mean":
                        ax.plot([s["n"] for s in points], [s["q_max_observed"] for s in points],
                                ":", color=color, label=f"{dist}: Qmax наблюд.")
                    ax.set(xlabel="N", ylabel=label)
                    ax.grid(alpha=0.25)
                    ax.legend(fontsize=8)
            axes[0, 1].axhline(2, color="black", linestyle="--", linewidth=0.8)
            fig.suptitle(f"CSMA 100 Мбит/с, 30 м, α={alpha}; интервалы ≥99% (приближённо)")
            fig.tight_layout()
            fig.savefig(self.out / f"alpha-{alpha}.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ns3", type=Path, help="Built ns-3 source tree; locate binary in build/scratch")
    source.add_argument("--binary", type=Path, help="Built csma-v3 executable")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--alphas", type=int, nargs="+", choices=[10, 50, 100], default=[10, 50, 100])
    parser.add_argument("--distributions", nargs="+", choices=["exp", "uniform"], default=["exp", "uniform"])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--search-runs", type=int, default=8)
    parser.add_argument("--min-runs", type=int, default=32)
    parser.add_argument("--max-runs", type=int, default=4096)
    parser.add_argument("--max-n", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300, help="Wall-clock timeout of one simulation, seconds")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent independent simulations")
    args = parser.parse_args()
    if not (2 <= args.min_runs <= args.max_runs < 1000000 and args.search_runs >= 2 and
            args.max_n >= 2 and 0 < args.seed < 4294944430 and args.timeout > 0 and args.jobs >= 1):
        parser.error("Invalid limits/seed")
    study = Study(args)
    cases = []
    for alpha in args.alphas:
        for dist in args.distributions:
            cases.append(study.run_case(dist, alpha))
            study.write_results(cases)  # Preserve completed cases even if a later run fails.
    incomplete = (any(not s["precise"] for s in study.summaries.values()) or
                  any(c[k]["status"] != "adjacent_checked" for c in cases for k in ("loss", "retries")))
    print(f"Tables, raw data and plots: {study.out}")
    if incomplete:
        print("NOT COMPLETE: inspect statuses; raise --max-runs/--max-n as appropriate.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
