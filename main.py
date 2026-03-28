#!/usr/bin/env python3
"""
AFIS Demo – Interactive command-line tool for fingerprint enrollment,
verification, and identification.
"""

import os
import sys
import time
import cv2
import numpy as np

# Import the AFIS system from our core module
from afis_core import AFISSystem

# ----------------------------------------------------------------------
# Helper functions for console output
# ----------------------------------------------------------------------
def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def print_warning(text):
    print(f"⚠️  {text}")

# ----------------------------------------------------------------------
# Main demo class
# ----------------------------------------------------------------------
class AFISDemo:
    def __init__(self, db_path="fingerprint_templates.db"):
        self.afis = AFISSystem(db_path)
        self.running = True

    def run(self):
        """Main interactive loop."""
        print_header("AFIS Fingerprint Recognition System")
        print("Welcome! This system allows you to:")
        print("  • Enroll new fingerprints")
        print("  • Verify a fingerprint against a claimed identity")
        print("  • Identify an unknown fingerprint")
        print("  • View system statistics")
        print("  • Delete a person's records")
        print("\nType 'help' for commands, 'quit' to exit.")

        while self.running:
            try:
                cmd = input("\n> ").strip().lower()
                if not cmd:
                    continue
                self.execute_command(cmd)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print_error(f"Unexpected error: {e}")

    def execute_command(self, cmd):
        """Route commands to appropriate methods."""
        if cmd == "quit" or cmd == "exit":
            self.running = False
            print_info("Exiting AFIS Demo.")
        elif cmd == "help":
            self.show_help()
        elif cmd == "enroll":
            self.enroll()
        elif cmd == "verify":
            self.verify()
        elif cmd == "identify":
            self.identify()
        elif cmd == "stats":
            self.show_stats()
        elif cmd == "delete":
            self.delete_person()
        else:
            print_error("Unknown command. Type 'help' for available commands.")

    def show_help(self):
        """Display available commands."""
        print("\nAvailable commands:")
        print("  enroll   - Enroll a new fingerprint")
        print("  verify   - Verify a fingerprint against a claimed identity")
        print("  identify - Identify an unknown fingerprint")
        print("  stats    - Show system statistics")
        print("  delete   - Delete all records for a person")
        print("  quit     - Exit the program")

    def get_image_path(self):
        """Prompt for a valid image file path."""
        while True:
            path = input("Enter image path: ").strip()
            if not path:
                continue
            if not os.path.isfile(path):
                print_error(f"File not found: {path}")
                continue
            # Check if it looks like an image (optional)
            ext = os.path.splitext(path)[1].lower()
            if ext not in ['.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg']:
                print_warning(f"Unusual extension '{ext}'. Proceed anyway? (y/n)")
                if input().lower() != 'y':
                    continue
            return path

    def enroll(self):
        """Enroll a new fingerprint."""
        print_header("ENROLLMENT")
        person_id = input("Enter person ID (e.g., 'john_doe'): ").strip()
        if not person_id:
            print_error("Person ID cannot be empty.")
            return

        print("Select finger position (optional, press Enter to skip):")
        print("  1 - Right thumb")
        print("  2 - Right index")
        print("  3 - Right middle")
        print("  4 - Right ring")
        print("  5 - Right little")
        print("  6 - Left thumb")
        print("  7 - Left index")
        print("  8 - Left middle")
        print("  9 - Left ring")
        print("  10 - Left little")
        pos_input = input("Position (1-10, or blank): ").strip()
        finger_pos = int(pos_input) if pos_input.isdigit() else 1

        image_path = self.get_image_path()
        if not image_path:
            return

        print(f"Processing image: {image_path} ...")
        template = self.afis.enroll(image_path, person_id, finger_pos)

        if template:
            print_success(f"Enrollment successful for {person_id}.")
            print(f"   Template ID: {template.fingerprint_id}")
            print(f"   Minutiae: {len(template.minutiae)}")
            print(f"   Pattern type: {template.global_features.get('pattern_type', 'unknown')}")
            # Optionally visualize
            if input("Show minutiae visualization? (y/n): ").lower() == 'y':
                self.afis.visualize_minutiae(image_path=image_path)
        else:
            print_error("Enrollment failed. Check image quality and try again.")

    def verify(self):
        """1:1 verification."""
        print_header("VERIFICATION")
        claimed_id = input("Enter claimed person ID: ").strip()
        if not claimed_id:
            print_error("Person ID cannot be empty.")
            return

        image_path = self.get_image_path()
        if not image_path:
            return

        print(f"Verifying {image_path} against {claimed_id} ...")
        is_match, score, details = self.afis.verify(image_path, claimed_id)

        if is_match:
            print_success(f"VERIFIED: The fingerprint matches {claimed_id} (score: {score:.3f})")
        else:
            print_warning(f"NOT VERIFIED: Score {score:.3f} below threshold.")
        print(f"  Matched minutiae: {details.get('matched_minutiae', '?')}")

    def identify(self):
        """1:N identification."""
        print_header("IDENTIFICATION")
        image_path = self.get_image_path()
        if not image_path:
            return

        top_k = input("Number of top candidates to show (default 5): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 5

        print(f"Identifying fingerprint...")
        results = self.afis.identify(image_path, top_k=top_k)

        if not results:
            print_error("Identification failed or database empty.")
            return

        print("\nTop candidates:")
        for i, (person_id, score, details) in enumerate(results, 1):
            print(f"  {i}. {person_id} – score: {score:.3f} (matches: {details.get('matched_minutiae', '?')})")

        # Optionally show the best match visualization
        if results and input("Show visualization of best match? (y/n): ").lower() == 'y':
            best_id = results[0][0]
            # Retrieve the reference template (we need the original image, which we don't have)
            # We can at least show the query minutiae
            self.afis.visualize_minutiae(image_path=image_path)

    def show_stats(self):
        """Display system statistics."""
        print_header("SYSTEM STATISTICS")
        stats = self.afis.get_statistics()
        print(f"Total templates: {stats.get('total_templates', 0)}")
        print(f"Unique persons:  {stats.get('unique_persons', 0)}")
        print(f"Average minutiae per template: {stats.get('avg_minutiae_count', 0)}")
        print("\nPattern type distribution:")
        for pat, cnt in stats.get('pattern_distribution', {}).items():
            print(f"  {pat}: {cnt}")

        # Preprocessing stats
        pre = stats.get('preprocessing_stats', {})
        if pre.get('processed_images', 0) > 0:
            print(f"\nPreprocessing stats:")
            print(f"  Images processed: {pre.get('processed_images', 0)}")
            print(f"  Avg processing time: {pre.get('avg_processing_time', 0):.3f} s")
            print(f"  Avg quality score: {pre.get('avg_quality_score', 0):.3f}")

        # Extractor stats
        ext = stats.get('extractor_stats', {})
        if ext.get('images_processed', 0) > 0:
            print(f"\nFeature extraction stats:")
            print(f"  Images processed: {ext.get('images_processed', 0)}")
            print(f"  Avg minutiae count: {ext.get('avg_minutiae', 0):.1f}")

        # Matcher stats
        mat = stats.get('matcher_stats', {})
        if mat.get('verifications', 0) > 0:
            print(f"\nMatching stats:")
            print(f"  Verifications performed: {mat.get('verifications', 0)}")
            print(f"  Identifications performed: {mat.get('identifications', 0)}")
            if 'avg_match_score' in mat:
                print(f"  Average match score: {mat.get('avg_match_score', 0):.3f}")

    def delete_person(self):
        """Delete all records for a person."""
        print_header("DELETE PERSON")
        person_id = input("Enter person ID to delete: ").strip()
        if not person_id:
            print_error("Person ID cannot be empty.")
            return
        confirm = input(f"Delete all records for '{person_id}'? (y/n): ").lower()
        if confirm == 'y':
            count = self.afis.delete_person(person_id)
            print_success(f"Deleted {count} template(s) for {person_id}.")
        else:
            print_info("Deletion cancelled.")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: If you want to create a synthetic test image for quick testing
    # you can uncomment the lines below.
    # from afis_core import FingerprintAcquisition
    # test_acq = FingerprintAcquisition()
    # test_img = test_acq.create_test_image('loop')
    # cv2.imwrite("test_loop.png", test_img)
    # print("Created test fingerprint: test_loop.png")

    demo = AFISDemo()
    demo.run()