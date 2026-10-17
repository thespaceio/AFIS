#!/usr/bin/env python3
"""
AFIS command-line interface.

Interactive menu (as in the original demo):
    python -m afis

One-shot commands (scriptable):
    python -m afis enroll  alice samples/101_1.tif --finger 2
    python -m afis verify  alice samples/101_2.tif
    python -m afis identify samples/101_2.tif --top 3
    python -m afis stats
    python -m afis delete  alice
"""

from __future__ import annotations

import argparse
import os
import sys

from .acquisition import AcquisitionError
from .database import TemplateIntegrityError
from .system import FINGER_NAMES, AFISSystem, EnrollmentError

DEFAULT_DB = os.environ.get("AFIS_DB", "afis_templates.db")


def ok(text: str) -> None:
    print(f"[+] {text}")


def err(text: str) -> None:
    print(f"[x] {text}")


def info(text: str) -> None:
    print(f"[i] {text}")


def header(text: str) -> None:
    print("\n" + "=" * 60 + f"\n {text}\n" + "=" * 60)


# ---------------------------------------------------------------------- actions
def do_enroll(afis: AFISSystem, person: str, path: str, finger: int = 0) -> bool:
    try:
        t = afis.enroll(path, person, finger)
    except (EnrollmentError, AcquisitionError, FileNotFoundError, ValueError) as e:
        err(f"Enrollment failed: {e}")
        return False
    g = t.global_features
    ok(f"Enrolled {person} ({FINGER_NAMES.get(finger, finger)})")
    print(f"    template id : {t.fingerprint_id}")
    print(f"    minutiae    : {len(t.minutiae)}")
    print(f"    pattern     : {g.get('pattern_type')}   quality: {g.get('quality'):.2f}")
    return True


def do_verify(afis: AFISSystem, person: str, path: str) -> bool:
    try:
        match, score, details = afis.verify(path, person)
    except (AcquisitionError, FileNotFoundError, ValueError, TemplateIntegrityError) as e:
        err(f"Verification failed: {e}")
        return False
    if "error" in details:
        err(details["error"])
        return False
    thr = afis.matcher.config.threshold
    if match:
        ok(f"VERIFIED as {person}  (score {score:.3f} >= threshold {thr:.2f})")
    else:
        err(f"NOT VERIFIED  (score {score:.3f} < threshold {thr:.2f})")
    print(f"    matched minutiae: {details.get('matched_minutiae', 0)}")
    return match


def do_identify(afis: AFISSystem, path: str, top: int = 5) -> None:
    try:
        results = afis.identify(path, top)
    except (AcquisitionError, FileNotFoundError, ValueError, TemplateIntegrityError) as e:
        err(f"Identification failed: {e}")
        return
    if not results:
        err("The database is empty. Enroll someone first.")
        return
    best_id, best_score, best = results[0]
    if best["is_match"]:
        ok(f"IDENTIFIED as {best_id}  (score {best_score:.3f})")
    else:
        err("No match above the threshold.")
    print("\n    Rank  Person               Score   Matched")
    for k, (pid, score, d) in enumerate(results, 1):
        flag = "*" if d["is_match"] else " "
        print(f"    {k:>3}{flag}  {pid:<20} {score:6.3f}   {d.get('matched_minutiae', 0)}")


def do_stats(afis: AFISSystem) -> None:
    s = afis.get_statistics()
    print(f"    templates        : {s['total_templates']}")
    print(f"    people           : {s['unique_persons']}")
    print(f"    avg minutiae     : {s['avg_minutiae_count']}")
    print(f"    match threshold  : {s['threshold']}")
    for pattern, n in s.get("pattern_distribution", {}).items():
        print(f"    pattern {pattern:<9}: {n}")


# ---------------------------------------------------------------------- interactive
def ask_path() -> str:
    while True:
        p = input("Image path: ").strip().strip('"').strip("'")
        if os.path.isfile(p):
            return p
        err(f"File not found: {p}")


def interactive(afis: AFISSystem) -> None:
    header("AFIS - Automated Fingerprint Identification System")
    print("Commands: enroll, verify, identify, stats, delete, help, quit")
    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "help":
            print("enroll | verify | identify | stats | delete | quit")
        elif cmd == "enroll":
            person = input("Person ID: ").strip()
            print("Finger: " + ", ".join(f"{k}={v}" for k, v in FINGER_NAMES.items()))
            pos = input("Finger number (Enter = 0): ").strip()
            do_enroll(afis, person, ask_path(), int(pos) if pos.isdigit() and int(pos) in FINGER_NAMES else 0)
        elif cmd == "verify":
            do_verify(afis, input("Claimed person ID: ").strip(), ask_path())
        elif cmd == "identify":
            do_identify(afis, ask_path())
        elif cmd == "stats":
            do_stats(afis)
        elif cmd == "delete":
            person = input("Person ID to delete: ").strip()
            if person and input(f"Delete all templates for '{person}'? (y/n): ").lower() == "y":
                ok(f"Deleted {afis.delete_person(person)} template(s)")
        elif cmd:
            err("Unknown command. Type 'help'.")


# ---------------------------------------------------------------------- entry
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="afis", description="Automated Fingerprint Identification System")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"database file (default {DEFAULT_DB})")
    sub = ap.add_subparsers(dest="command")
    e = sub.add_parser("enroll", help="add a fingerprint")
    e.add_argument("person")
    e.add_argument("image")
    e.add_argument("--finger", type=int, default=0, choices=range(0, 11))
    v = sub.add_parser("verify", help="1:1 check against a claimed identity")
    v.add_argument("person")
    v.add_argument("image")
    i = sub.add_parser("identify", help="1:N search of the database")
    i.add_argument("image")
    i.add_argument("--top", type=int, default=5)
    sub.add_parser("stats", help="database statistics")
    d = sub.add_parser("delete", help="delete all templates of a person")
    d.add_argument("person")
    args = ap.parse_args(argv)

    afis = AFISSystem(args.db)
    try:
        if args.command == "enroll":
            return 0 if do_enroll(afis, args.person, args.image, args.finger) else 1
        if args.command == "verify":
            return 0 if do_verify(afis, args.person, args.image) else 1
        if args.command == "identify":
            do_identify(afis, args.image, args.top)
        elif args.command == "stats":
            do_stats(afis)
        elif args.command == "delete":
            ok(f"Deleted {afis.delete_person(args.person)} template(s)")
        else:
            interactive(afis)
        return 0
    finally:
        afis.close()


if __name__ == "__main__":
    sys.exit(main())
