#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rake.py — Анализ влияния рейка на винрейт и EV/час
"""

import argparse
import json
import csv
import math
import sys
from statistics import mean

# ===============================
# Вспомогательные функции
# ===============================

def round_up_to_step(x: float, step: float) -> float:
    """Округление вверх до ближайшего шага."""
    if step <= 0:
        return x
    return math.ceil(x / step) * step


def estimate_player_share(seats: int, vpip: float = None, pfr: float = None) -> float:
    """Приближение доли рейка, которую платит игрок."""
    if vpip is not None and pfr is not None:
        share = max(0.15, 0.5 * (vpip + pfr))
    else:
        share = 0.33 if seats == 6 else 0.25
    return min(share, 0.7)


def default_hands_per_hour(fmt: str, seats: int) -> int:
    """Возвращает дефолтное число рук в час по формату и посадке."""
    fmt = fmt.lower()
    if fmt == "cash":
        if seats == 6:
            return 600
        elif seats == 9:
            return 450
    elif fmt == "zoom":
        if seats == 6:
            return 800
        elif seats == 9:
            return 700
    return 600


def compute_metrics(params: dict) -> dict:
    """
    Расчёт всех основных метрик рейка и EV.
    Возвращает словарь с результатами.
    """
    avg_pot_bb = params["avg_pot_bb"]
    rake_percent = params["rake_percent"]
    rake_cap_bb = params["rake_cap_bb"]
    rake_step = params["rake_step"]
    triggers_frac = params["rake_triggers_frac"]
    winrate_bb100 = params["winrate_bb100"]
    hands_per_hour = params["hands_per_hour"]
    bb_value = params.get("bb_value")
    hours = params.get("hours", None)
    seats = params["seats"]
    vpip = params.get("vpip")
    pfr = params.get("pfr")

    # Расчёт рейка на раздачу
    rake_raw = avg_pot_bb * rake_percent / 100.0
    rake_capped = min(rake_raw, rake_cap_bb)
    rake_final = round_up_to_step(rake_capped, rake_step)
    rake_per_100_bb = 100.0 * triggers_frac * rake_final

    # Доля игрока
    share = estimate_player_share(seats, vpip, pfr)
    player_rake_bb100 = rake_per_100_bb * share

    net_bb100 = winrate_bb100 - player_rake_bb100
    ev_per_hour_bb = net_bb100 * hands_per_hour / 100.0

    result = dict(
        rake_raw=rake_raw,
        rake_capped=rake_capped,
        rake_final=rake_final,
        rake_per_100_bb=rake_per_100_bb,
        player_share=share,
        player_rake_bb100=player_rake_bb100,
        net_bb100=net_bb100,
        ev_per_hour_bb=ev_per_hour_bb,
    )

    # Валютные метрики
    if bb_value:
        ev_per_hour_$ = ev_per_hour_bb * bb_value
        result["ev_per_hour_$"] = ev_per_hour_$
        if hours:
            result["ev_per_session_$"] = ev_per_hour_$ * hours

    return result


def print_summary(params: dict, result: dict):
    """Вывод краткого отчёта в консоль."""
    print("=" * 60)
    print(f"Формат: {params['format']}  |  Seats: {params['seats']}  |  Hands/h: {params['hands_per_hour']}")
    print(f"Player share (оценка): {result['player_share']:.2f}")
    print("-" * 60)
    print(f"Winrate (pre-rake):     {params['winrate_bb100']:+.2f} bb/100")
    print(f"Rake per 100 (table):   {result['rake_per_100_bb']:.2f} bb/100")
    print(f"Your share (bb/100):    {result['player_rake_bb100']:.2f}")
    print("-" * 60)
    print(f"Net winrate:            {result['net_bb100']:+.2f} bb/100")
    print(f"EV/h (bb):              {result['ev_per_hour_bb']:+.2f}")
    if "ev_per_hour_$" in result:
        print(f"EV/h ($):               {result['ev_per_hour_$']:+.2f}")
    if "ev_per_session_$" in result:
        print(f"EV/session ($):         {result['ev_per_session_$']:+.2f}")
    print("=" * 60)


def write_csv(rows, path):
    """Сохраняет результаты в CSV."""
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Сохранено: {path}")


def write_md(result: dict, path: str):
    """Создаёт markdown-отчёт."""
    lines = []
    lines.append("# Rake Report\n")
    lines.append("## Входные параметры\n")
    for k, v in result["input"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Результаты\n")
    for k, v in result["output"].items():
        lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, (int, float)) else f"- **{k}**: {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[MD] Сохранено: {path}")


# ===============================
# CLI-команды
# ===============================

def cmd_single(args):
    params = {
        "format": args.format,
        "seats": args.seats,
        "bb_value": args.bb,
        "winrate_bb100": args.winrate_bb100,
        "hands_per_hour": args.hands_per_hour or default_hands_per_hour(args.format, args.seats),
        "rake_percent": args.rake_percent,
        "rake_cap_bb": args.rake_cap_bb,
        "rake_step": args.rake_step,
        "rake_triggers_frac": args.rake_triggers_frac,
        "avg_pot_bb": args.avg_pot_bb,
        "vpip": args.vpip,
        "pfr": args.pfr,
        "hours": args.hours,
    }

    result = compute_metrics(params)
    print_summary(params, result)

    if args.csv_out:
        write_csv([dict(input=params, **result)], args.csv_out)
    if args.md_out:
        write_md({"input": params, "output": result}, args.md_out)


def cmd_compare(args):
    """Сравнение нескольких лимитов."""
    rows = []
    if args.table:
        with open(args.table, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    else:
        # Пример: "bb=[0.5,1,2];rake_percent=[5,5,5];cap_bb=[3,2,1]"
        parts = [p.strip() for p in args.grid.split(";") if p.strip()]
        data = []
        base = {}
        for p in parts:
            key, val = p.split("=")
            key = key.strip()
            val = val.strip()
            if val.startswith("["):
                base[key] = eval(val)
            else:
                base[key] = [float(val)]
        # Транспонируем для каждой позиции списка
        n = len(list(base.values())[0])
        for i in range(n):
            entry = {k: v[i] if i < len(v) else v[-1] for k, v in base.items()}
            data.append(entry)

    for row in data:
        params = {
            "format": args.format,
            "seats": args.seats,
            "bb_value": float(row.get("bb", args.bb or 1)),
            "winrate_bb100": float(row.get("winrate_bb100", args.winrate_bb100)),
            "hands_per_hour": int(row.get("hands_per_hour", args.hands_per_hour or default_hands_per_hour(args.format, args.seats))),
            "rake_percent": float(row.get("rake_percent", args.rake_percent)),
            "rake_cap_bb": float(row.get("cap_bb", args.rake_cap_bb)),
            "rake_step": args.rake_step,
            "rake_triggers_frac": float(row.get("rake_triggers_frac", args.rake_triggers_frac)),
            "avg_pot_bb": float(row.get("avg_pot_bb", args.avg_pot_bb)),
            "vpip": args.vpip,
            "pfr": args.pfr,
        }
        result = compute_metrics(params)
        merged = {"limit": row.get("name", f"L{i+1}") if isinstance(row, dict) else f"L{i+1}", **params, **result}
        rows.append(merged)

    # Сортировка
    key = "ev_per_hour_$" if args.bb else "net_bb100"
    rows.sort(key=lambda r: r.get(key, 0), reverse=True)

    print(f"\n{'Limit':<10} | Net bb/100 | EV/h bb | EV/h $")
    print("-" * 50)
    for r in rows:
        print(f"{r['limit']:<10} | {r['net_bb100']:+6.2f} | {r['ev_per_hour_bb']:+7.2f} | {r.get('ev_per_hour_$', 0):+7.2f}")
    print("-" * 50)

    if args.csv_out:
        write_csv(rows, args.csv_out)


def cmd_sensitivity(args):
    """Генерация таблицы чувствительности по двум параметрам."""
    def parse_axis(s):
        name, rng = s.split(":")[0], s.split(":")[1:]
        return name, float(rng[0]), float(rng[1]), float(rng[2])

    x_name, x_min, x_max, x_step = parse_axis(args.x)
    y_name, y_min, y_max, y_step = parse_axis(args.y)

    x_vals = [round(x_min + i * x_step, 5) for i in range(int((x_max - x_min) / x_step) + 1)]
    y_vals = [round(y_min + i * y_step, 5) for i in range(int((y_max - y_min) / y_step) + 1)]

    results = []
    for xv in x_vals:
        row = []
        for yv in y_vals:
            params = {
                "format": args.format,
                "seats": args.seats,
                "bb_value": args.bb,
                "winrate_bb100": args.winrate_bb100,
                "hands_per_hour": args.hands_per_hour or default_hands_per_hour(args.format, args.seats),
                "rake_percent": args.rake_percent,
                "rake_cap_bb": args.rake_cap_bb,
                "rake_step": args.rake_step,
                "rake_triggers_frac": yv if y_name == "rake_triggers_frac" else args.rake_triggers_frac,
                "avg_pot_bb": xv if x_name == "avg_pot_bb" else args.avg_pot_bb,
                "vpip": args.vpip,
                "pfr": args.pfr,
            }
            out = compute_metrics(params)
            val = out.get(args.keep, 0)
            row.append(val)
            results.append(dict(x=xv, y=yv, metric=val))
        print(" ".join(f"{v:7.2f}" for v in row))

    if args.csv_out:
        write_csv(results, args.csv_out)


def cmd_selftest():
    """Минимальный набор самотестов."""
    print("== SELFTEST ==")
    # 1. Округление
    assert round_up_to_step(2.01, 0.25) == 2.25
    # 2. Кэп
    p = {
        "format": "cash", "seats": 6, "avg_pot_bb": 100, "rake_percent": 5,
        "rake_cap_bb": 3, "rake_step": 0.01, "rake_triggers_frac": 0.5,
        "winrate_bb100": 5, "hands_per_hour": 600
    }
    r = compute_metrics(p)
    assert math.isclose(r["rake_final"], 3.00)
    # 3. Случай без ошибок
    p["avg_pot_bb"] = 10
    r = compute_metrics(p)
    assert r["net_bb100"] != 0
    print("OK ✅")


# ===============================
# Основной вход
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Анализ влияния рейка на винрейт и EV/час")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Подкоманда: single
    s = subparsers.add_parser("single", help="Расчёт для одного пула/лимита")
    s.add_argument("--format", default="cash", choices=["cash", "zoom"])
    s.add_argument("--seats", type=int, default=6)
    s.add_argument("--bb", dest="bb", type=float, default=None)
    s.add_argument("--winrate_bb100", type=float, required=True)
    s.add_argument("--hands_per_hour", type=int, default=None)
    s.add_argument("--rake_percent", type=float, required=True)
    s.add_argument("--rake_cap_bb", type=float, required=True)
    s.add_argument("--rake_step", type=float, default=0.01)
    s.add_argument("--rake_triggers_frac", type=float, required=True)
    s.add_argument("--avg_pot_bb", type=float, required=True)
    s.add_argument("--vpip", type=float, default=None)
    s.add_argument("--pfr", type=float, default=None)
    s.add_argument("--hours", type=float, default=None)
    s.add_argument("--csv_out", default=None)
    s.add_argument("--md_out", default=None)
    s.set_defaults(func=cmd_single)

    # Подкоманда: compare
    c = subparsers.add_parser("compare", help="Сравнение нескольких лимитов")
    c.add_argument("--format", default="cash", choices=["cash", "zoom"])
    c.add_argument("--seats", type=int, default=6)
    c.add_argument("--bb", type=float, default=None)
    c.add_argument("--winrate_bb100", type=float, required=True)
    c.add_argument("--hands_per_hour", type=int, default=None)
    c.add_argument("--rake_percent", type=float, required=True)
    c.add_argument("--rake_cap_bb", type=float, required=True)
    c.add_argument("--rake_step", type=float, default=0.01)
    c.add_argument("--rake_triggers_frac", type=float, required=True)
    c.add_argument("--avg_pot_bb", type=float, required=True)
    c.add_argument("--vpip", type=float, default=None)
    c.add_argument("--pfr", type=float, default=None)
    c.add_argument("--grid", type=str, default=None)
    c.add_argument("--table", type=str, default=None)
    c.add_argument("--csv_out", default=None)
    c.set_defaults(func=cmd_compare)

    # Подкоманда: sensitivity
    sen = subparsers.add_parser("sensitivity", help="Таблица чувствительности (X,Y)")
    sen.add_argument("--format", default="cash", choices=["cash", "zoom"])
    sen.add_argument("--seats", type=int, default=6)
    sen.add_argument("--bb", type=float, default=None)
    sen.add_argument("--winrate_bb100", type=float, required=True)
    sen.add_argument("--hands_per_hour", type=int, default=None)
    sen.add_argument("--rake_percent", type=float, required=True)
    sen.add_argument("--rake_cap_bb", type=float, required=True)
    sen.add_argument("--rake_step", type=float, default=0.01)
    sen.add_argument("--rake_triggers_frac", type=float, required=True)
    sen.add_argument("--avg_pot_bb", type=float, required=True)
    sen.add_argument("--vpip", type=float, default=None)
    sen.add_argument("--pfr", type=float, default=None)
    sen.add_argument("--x", type=str, required=True)
    sen.add_argument("--y", type=str, required=True)
    sen.add_argument("--keep", type=str, default="net_bb100")
    sen.add_argument("--csv_out", default=None)
    sen.set_defaults(func=cmd_sensitivity)

    # Подкоманда: selftest
    st = subparsers.add_parser("selftest", help="Запуск встроенных тестов")
    st.set_defaults(func=lambda _: cmd_selftest())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(2)