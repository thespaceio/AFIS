#!/usr/bin/env python3
"""
AFIS command-line interface.

Interactive menu:
    afis                       (or: python -m afis)

One-shot commands (scriptable):
    afis enroll  alice print1.png --finger 2
    afis verify  alice print2.png          exit code 0 = verified, 1 = not verified
    afis identify print2.png --top 3       exit code 0 = identified, 1 = no match
    afis analyse print1.png
    afis list | stats | audit | check
    afis demo                              enrol the bundled synthetic demo prints
    afis delete alice                      asks for confirmation (or --yes)
    afis delete-all                        asks you to type DELETE (or --yes)
    afis info                              where the database, key and settings are

Exit codes: 0 success, 1 negative result (no match), 2 error.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from . import __version__
from .acquisition import AcquisitionError
from .config import Settings, default_db_path
from .database import AFISKeyError, TemplateIntegrityError, key_path_for
from .system import FINGER_NAMES, AFISSystem, EnrollmentError

DISCLAIMER = ("Match scores are probabilistic. AFIS is an educational system and has not been "
              "validated for forensic, legal or law-enforcement decisions.")

USER_ERRORS = (EnrollmentError, AcquisitionError, FileNotFoundError, ValueError, TemplateIntegrityError)


def ok(text: str) -> None:
    print(f"[+] {text}")


def err(text: str) -> None:
    print(f"[x] {text}", file=sys.stderr)


def info(text: str) -> None:
    print(f"[i] {text}")


def header(text: str) -> None:
    print("\n" + "=" * 60 + f"\n {text}\n" + "=" * 60)


def confirm(prompt: str, expected: str = "y") -> bool:
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.lower() == expected.lower() if expected == "y" else answer == expected


# ---------------------------------------------------------------------- actions
def do_enroll(afis: AFISSystem, person: str, path: str, finger: int = 0) -> bool:
    try:
        t = afis.enroll(path, person, finger)
    except USER_ERRORS as e:
        err(f"Enrollment failed: {e}")
        return False
    g = t.global_features
    ok(f"Enrolled {t.person_id} ({FINGER_NAMES.get(finger, finger)})")
    print(f"    template id : {t.fingerprint_id}")
    print(f"    minutiae    : {len(t.minutiae)}")
    print(f"    pattern     : {g.get('pattern_type')}   quality: {g.get('quality'):.2f}")
    return True


def do_verify(afis: AFISSystem, person: str, path: str):
    """True = verified, False = not verified, None = error."""
    try:
        match, score, details = afis.verify(path, person)
    except USER_ERRORS as e:
        err(f"Verification failed: {e}")
        return None
    if "error" in details:
        err(details["error"])
        return None
    thr = afis.threshold
    if match:
        ok(f"MATCH: consistent with {person}  (score {score:.3f} >= threshold {thr:.2f})")
    else:
        err(f"NO MATCH for {person}  (score {score:.3f} < threshold {thr:.2f})")
    print(f"    matched minutiae: {details.get('matched_minutiae', 0)}")
    return match


def do_identify(afis: AFISSystem, path: str, top: int = 5):
    """True = identified, False = no match, None = error / empty database."""
    try:
        results = afis.identify(path, top)
    except USER_ERRORS as e:
        err(f"Identification failed: {e}")
        return None
    if not results:
        err("The database is empty. Enroll someone first (or run: afis demo).")
        return None
    best_id, best_score, best = results[0]
    if best["is_match"]:
        ok(f"Best candidate: {best_id}  (score {best_score:.3f})")
    else:
        err("No candidate scored above the threshold.")
    print("\n    Rank  Person               Score   Matched")
    for k, (pid, score, d) in enumerate(results, 1):
        flag = "*" if d["is_match"] else " "
        print(f"    {k:>3}{flag}  {pid:<20} {score:6.3f}   {d.get('matched_minutiae', 0)}")
    return bool(best["is_match"])


def do_analyse(afis: AFISSystem, path: str) -> bool:
    try:
        p = afis.process(path)
    except USER_ERRORS as e:
        err(f"Analysis failed: {e}")
        return False
    endings = sum(m.minutiae_type == "ending" for m in p.minutiae)
    g = p.global_features
    print(f"    size          : {p.image.shape[1]} x {p.image.shape[0]} px")
    print(f"    quality       : {p.result.quality:.2f}")
    print(f"    minutiae      : {len(p.minutiae)} ({endings} endings, {len(p.minutiae) - endings} bifurcations)")
    print(f"    pattern       : {g['pattern_type']} ({len(g['core_points'])} core, {len(g['delta_points'])} delta)")
    print(f"    ridge spacing : {p.result.median_wavelength:.1f} px")
    print(f"    time          : {p.result.timings['total'] * 1000:.0f} ms")
    return True


def do_stats(afis: AFISSystem) -> None:
    s = afis.get_statistics()
    print(f"    templates        : {s['total_templates']}")
    print(f"    people           : {s['unique_persons']}")
    print(f"    avg minutiae     : {s['avg_minutiae_count']}")
    print(f"    match threshold  : {s['threshold']:.2f}")
    for pattern, n in s.get("pattern_distribution", {}).items():
        print(f"    pattern {pattern or '?':<9}: {n}")


def do_list(afis: AFISSystem) -> None:
    rows = afis.database.list_templates()
    if not rows:
        info("No templates enrolled.")
        return
    print(f"    {'Person':<22}{'Finger':<14}{'Pattern':<9}{'Min.':>5}  {'Quality':>7}  Enrolled (UTC)")
    for r in rows:
        print(f"    {r['person_id']:<22}{FINGER_NAMES.get(r['finger_position'], '?'):<14}"
              f"{(r['pattern_type'] or ''):<9}{r['minutiae_count']:>5}  {(r['quality'] or 0):>7.2f}  "
              f"{(r['created_at'] or '')[:16].replace('T', ' ')}")


def do_audit(afis: AFISSystem, limit: int, export: str = None) -> None:
    entries = afis.database.audit_entries(limit)
    if export:
        with open(export, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "timestamp", "action", "person_id", "fingerprint_id", "details"])
            w.writeheader()
            w.writerows(reversed(entries))
        ok(f"Exported {len(entries)} audit entries to {export}")
        return
    for e in entries:
        print(f"    {e['timestamp']}  {e['action']:<13} {e['person_id'] or '-':<20} {e['details'] or ''}")


def do_check(afis: AFISSystem) -> bool:
    res = afis.check_integrity()
    t, a = res["templates"], res["audit"]
    good = not t["bad"] and a["ok"]
    (ok if not t["bad"] else err)(f"Templates: {t['checked'] - len(t['bad'])}/{t['checked']} decrypted and intact")
    for fid, reason in t["bad"]:
        print(f"    {fid}: {reason}")
    if a["ok"]:
        ok(f"Audit log: chain of {a['checked']} entries intact" +
           (f" ({a['legacy']} older entries predate the chain)" if a["legacy"] else ""))
    elif a.get("truncated"):
        err("Audit log: entries were removed from the end of the log")
    else:
        err(f"Audit log: chain broken at entry {a['first_bad_id']} (edited, inserted or deleted entries)")
    return good


def do_info(db_path: str) -> None:
    key_env = bool(os.environ.get("AFIS_KEY"))
    print(f"    version   : {__version__}")
    print(f"    database  : {db_path}")
    print(f"    key       : {'AFIS_KEY environment variable' if key_env else key_path_for(db_path)}")
    print(f"    settings  : {Path(db_path).parent / 'settings.json'}")
    info("Back up the database AND the key together. Without the key the templates can't be read.")


def do_demo(afis: AFISSystem) -> None:
    from .demo import demo_dir
    people = afis.enroll_demo()
    ok(f"Enrolled {len(people)} synthetic demo people: {', '.join(people)} (impression 1 of each)")
    info(f"Try:  afis identify \"{demo_dir() / 'demo-03_2.png'}\"")
    info(f"      afis verify demo-01 \"{demo_dir() / 'demo-01_3.png'}\"")


# ---------------------------------------------------------------------- interactive
def ask_path() -> str:
    while True:
        p = input("Image path: ").strip().strip('"').strip("'")
        if os.path.isfile(p):
            return p
        err(f"File not found: {p}")


def interactive(afis: AFISSystem) -> None:
    header(f"AFIS {__version__} - Automated Fingerprint Identification System")
    print(DISCLAIMER)
    print("Commands: enroll, verify, identify, analyse, list, stats, audit, check, demo, delete, help, quit")
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "help":
                print("enroll | verify | identify | analyse | list | stats | audit | check | demo | delete | quit")
            elif cmd == "enroll":
                person = input("Person ID: ").strip()
                print("Finger: " + ", ".join(f"{k}={v}" for k, v in FINGER_NAMES.items()))
                pos = input("Finger number (Enter = 0): ").strip()
                do_enroll(afis, person, ask_path(), int(pos) if pos.isdigit() and int(pos) in FINGER_NAMES else 0)
            elif cmd == "verify":
                do_verify(afis, input("Claimed person ID: ").strip(), ask_path())
            elif cmd == "identify":
                do_identify(afis, ask_path())
            elif cmd in ("analyse", "analyze"):
                do_analyse(afis, ask_path())
            elif cmd == "list":
                do_list(afis)
            elif cmd == "stats":
                do_stats(afis)
            elif cmd == "audit":
                do_audit(afis, 30)
            elif cmd == "check":
                do_check(afis)
            elif cmd == "demo":
                do_demo(afis)
            elif cmd == "delete":
                person = input("Person ID to delete: ").strip()
                if person and confirm(f"Delete all templates for '{person}'? (y/n): "):
                    ok(f"Deleted {afis.delete_person(person)} template(s)")
            elif cmd:
                err("Unknown command. Type 'help'.")
        except (EOFError, KeyboardInterrupt):
            print()
            break


# ---------------------------------------------------------------------- entry
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="afis", description="Automated Fingerprint Identification System",
                                 epilog=DISCLAIMER)
    ap.add_argument("--version", action="version", version=f"AFIS {__version__}")
    ap.add_argument("--db", default=None, help="database file (default: the per-user AFIS data folder)")
    ap.add_argument("--threshold", type=float, default=None, help="match threshold for this run")
    sub = ap.add_subparsers(dest="command")
    e = sub.add_parser("enroll", help="add a fingerprint")
    e.add_argument("person")
    e.add_argument("image")
    e.add_argument("--finger", type=int, default=0, choices=range(0, 11), metavar="0-10")
    v = sub.add_parser("verify", help="1:1 check against a claimed identity")
    v.add_argument("person")
    v.add_argument("image")
    i = sub.add_parser("identify", help="1:N search of the database")
    i.add_argument("image")
    i.add_argument("--top", type=int, default=5)
    a = sub.add_parser("analyse", aliases=["analyze"], help="quality, minutiae and pattern of one image")
    a.add_argument("image")
    sub.add_parser("list", help="list enrolled templates")
    sub.add_parser("stats", help="database statistics")
    au = sub.add_parser("audit", help="show or export the audit log")
    au.add_argument("--limit", type=int, default=50)
    au.add_argument("--export", metavar="CSV", help="write the log to a CSV file")
    sub.add_parser("check", help="verify every template and the audit chain")
    sub.add_parser("demo", help="enrol the bundled synthetic demo prints")
    sub.add_parser("info", help="show where data, key and settings are stored")
    d = sub.add_parser("delete", help="delete all templates of a person")
    d.add_argument("person")
    d.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    da = sub.add_parser("delete-all", help="delete every template (the audit log is kept)")
    da.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db or default_db_path()
    if args.command == "info":
        do_info(db_path)
        return 0

    settings = Settings.load(Path(db_path).parent)
    try:
        afis = AFISSystem(db_path, store_images=settings.store_images,
                          threshold=args.threshold if args.threshold is not None else settings.threshold)
    except AFISKeyError as e:
        err(str(e))
        return 2
    except ValueError as e:
        err(str(e))
        return 2

    try:
        cmd = args.command
        if cmd == "enroll":
            return 0 if do_enroll(afis, args.person, args.image, args.finger) else 2
        if cmd == "verify":
            r = do_verify(afis, args.person, args.image)
            return 2 if r is None else (0 if r else 1)
        if cmd == "identify":
            r = do_identify(afis, args.image, args.top)
            return 2 if r is None else (0 if r else 1)
        if cmd in ("analyse", "analyze"):
            return 0 if do_analyse(afis, args.image) else 2
        if cmd == "list":
            do_list(afis)
        elif cmd == "stats":
            do_stats(afis)
        elif cmd == "audit":
            do_audit(afis, args.limit, args.export)
        elif cmd == "check":
            return 0 if do_check(afis) else 1
        elif cmd == "demo":
            do_demo(afis)
        elif cmd == "delete":
            if not (args.yes or confirm(f"Delete all templates for '{args.person}'? (y/n): ")):
                info("Cancelled.")
                return 1
            n = afis.delete_person(args.person)
            (ok if n else info)(f"Deleted {n} template(s) for '{args.person}'")
        elif cmd == "delete-all":
            n_total = afis.get_statistics()["total_templates"]
            if not args.yes and not confirm(f"This deletes all {n_total} template(s). Type DELETE to confirm: ",
                                            expected="DELETE"):
                info("Cancelled.")
                return 1
            ok(f"Deleted {afis.delete_all('DELETE')} template(s). The audit log was kept.")
        else:
            interactive(afis)
        return 0
    except TemplateIntegrityError as e:
        err(f"Integrity error: {e}")
        return 2
    finally:
        afis.close()


if __name__ == "__main__":
    sys.exit(main())
