from features import Minutia
import cv2
import numpy as np
import sqlite3
import pickle
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
import warnings



# ============================================================================
# 5. FINGERPRINT TEMPLATE
# ============================================================================

@dataclass
class FingerprintTemplate:
    """
    Compact representation of a fingerprint for storage and matching.
    Stores both local minutiae and global features.
    """
    fingerprint_id: str  # Unique identifier for this template
    person_id: str  # Owner of the fingerprint
    minutiae: List[Minutia]  # List of minutia points
    global_features: Dict[str, Any]  # Pattern type, core/delta, ridge density, etc.
    created_at: str = None  # ISO timestamp
    template_hash: str = None  # Hash for quick comparison (computed automatically)

    def __post_init__(self):
        """Generate creation timestamp and template hash if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.template_hash is None:
            self.template_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """
        Compute a hash of the template for fast equality checks.
        Based on sorted minutiae positions and global pattern type.
        """
        # Sort minutiae by x,y to get stable representation
        sorted_minutiae = sorted(self.minutiae, key=lambda m: (m.y, m.x))

        # Build a string representation
        data = []
        # Global features: pattern type + ridge density rounded
        data.append(self.global_features.get('pattern_type', 'unknown'))
        data.append(f"{self.global_features.get('ridge_density', 0):.2f}")
        data.append(str(len(self.minutiae)))

        # Add up to 50 minutiae (x, y, type, orientation quantized)
        for m in sorted_minutiae[:50]:
            # Quantize coordinates to reduce sensitivity
            qx = m.x // 5
            qy = m.y // 5
            qorient = int(m.orientation * 10) % 360
            data.append(f"{qx},{qy},{m.minutiae_type[0]},{qorient}")

        # Create SHA-256 hash
        hash_input = "|".join(data)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            'fingerprint_id': self.fingerprint_id,
            'person_id': self.person_id,
            'minutiae': [m.to_tuple() for m in self.minutiae],
            'global_features': self.global_features,
            'created_at': self.created_at,
            'template_hash': self.template_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FingerprintTemplate':
        """Reconstruct template from dictionary."""
        minutiae = [Minutia.from_tuple(t) for t in data['minutiae']]
        return cls(
            fingerprint_id=data['fingerprint_id'],
            person_id=data['person_id'],
            minutiae=minutiae,
            global_features=data['global_features'],
            created_at=data.get('created_at'),
            template_hash=data.get('template_hash')
        )

    def serialize(self) -> bytes:
        """Serialize template to bytes using pickle."""
        return pickle.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, data: bytes) -> 'FingerprintTemplate':
        """Deserialize template from bytes."""
        return cls.from_dict(pickle.loads(data))

    def get_minutiae_count(self) -> int:
        """Return number of minutiae."""
        return len(self.minutiae)

    def get_pattern_type(self) -> str:
        """Return pattern type (arch, loop, whorl, etc.)."""
        return self.global_features.get('pattern_type', 'unknown')


# ============================================================================
# 6. TEMPLATE DATABASE
# ============================================================================

class TemplateDatabase:
    """
    SQLite database for storing and retrieving fingerprint templates.
    Provides basic CRUD operations and indexing for faster retrieval.
    """

    def __init__(self, db_path: str = 'fingerprint_templates.db'):
        """
        Initialize database connection and create tables if they don't exist.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._create_tables()
        self._create_indexes()

    def _create_tables(self):
        """Create the necessary tables."""
        cursor = self.conn.cursor()

        # Templates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                fingerprint_id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                template_data BLOB NOT NULL,
                pattern_type TEXT,
                minutiae_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Optional: audit log for sensitive operations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                fingerprint_id TEXT,
                person_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        ''')

        self.conn.commit()

    def _create_indexes(self):
        """Create indexes for faster queries."""
        cursor = self.conn.cursor()

        # Index by person_id (for retrieving all templates of a person)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_person_id 
            ON templates (person_id)
        ''')

        # Index by pattern type (for filtering during identification)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_pattern_type 
            ON templates (pattern_type)
        ''')

        # Index by minutiae count (for additional filtering)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_minutiae_count 
            ON templates (minutiae_count)
        ''')

        self.conn.commit()

    def _log_audit(self, action: str, fingerprint_id: str = None,
                   person_id: str = None, details: str = None):
        """Internal method to log actions for security auditing."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO audit_log (action, fingerprint_id, person_id, details)
            VALUES (?, ?, ?, ?)
        ''', (action, fingerprint_id, person_id, details))
        self.conn.commit()

    def save_template(self, template: FingerprintTemplate) -> bool:
        """
        Save or update a fingerprint template in the database.

        Args:
            template: FingerprintTemplate object

        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()

            # Extract derived fields for indexing
            pattern_type = template.global_features.get('pattern_type', 'unknown')
            minutiae_count = len(template.minutiae)

            # Serialize template
            serialized = template.serialize()

            cursor.execute('''
                INSERT OR REPLACE INTO templates
                (fingerprint_id, person_id, template_data, pattern_type, minutiae_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                template.fingerprint_id,
                template.person_id,
                serialized,
                pattern_type,
                minutiae_count,
                template.created_at
            ))

            self.conn.commit()
            self._log_audit('SAVE', template.fingerprint_id, template.person_id,
                            f"Minutiae count: {minutiae_count}, Pattern: {pattern_type}")
            return True

        except Exception as e:
            print(f"Error saving template: {e}")
            return False

    def load_template(self, fingerprint_id: str) -> Optional[FingerprintTemplate]:
        """
        Load a single template by its ID.

        Args:
            fingerprint_id: Unique identifier of the template

        Returns:
            FingerprintTemplate object or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT template_data FROM templates WHERE fingerprint_id = ?
            ''', (fingerprint_id,))

            row = cursor.fetchone()
            if row:
                template = FingerprintTemplate.deserialize(row['template_data'])
                # Update last_accessed
                self._update_last_accessed(fingerprint_id)
                self._log_audit('LOAD', fingerprint_id, template.person_id)
                return template
            return None

        except Exception as e:
            print(f"Error loading template: {e}")
            return None

    def load_templates_by_person(self, person_id: str) -> List[FingerprintTemplate]:
        """
        Load all templates belonging to a specific person.

        Args:
            person_id: Person identifier

        Returns:
            List of FingerprintTemplate objects
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT template_data FROM templates WHERE person_id = ?
            ''', (person_id,))

            templates = []
            for row in cursor.fetchall():
                template = FingerprintTemplate.deserialize(row['template_data'])
                templates.append(template)
                self._update_last_accessed(template.fingerprint_id)

            return templates

        except Exception as e:
            print(f"Error loading templates for person: {e}")
            return []

    def load_all_templates(self) -> List[FingerprintTemplate]:
        """
        Load all templates from the database.
        Warning: May be memory-intensive for large databases.

        Returns:
            List of all FingerprintTemplate objects
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT template_data FROM templates')

            templates = []
            for row in cursor.fetchall():
                template = FingerprintTemplate.deserialize(row['template_data'])
                templates.append(template)

            return templates

        except Exception as e:
            print(f"Error loading all templates: {e}")
            return []

    def delete_template(self, fingerprint_id: str) -> bool:
        """
        Delete a template by its ID.

        Args:
            fingerprint_id: Unique identifier

        Returns:
            True if deleted, False otherwise
        """
        try:
            cursor = self.conn.cursor()

            # Get person_id for audit before deletion
            cursor.execute('SELECT person_id FROM templates WHERE fingerprint_id = ?',
                           (fingerprint_id,))
            row = cursor.fetchone()
            person_id = row['person_id'] if row else None

            cursor.execute('DELETE FROM templates WHERE fingerprint_id = ?', (fingerprint_id,))
            self.conn.commit()

            self._log_audit('DELETE', fingerprint_id, person_id)
            return True

        except Exception as e:
            print(f"Error deleting template: {e}")
            return False

    def delete_person_templates(self, person_id: str) -> int:
        """
        Delete all templates belonging to a person.

        Args:
            person_id: Person identifier

        Returns:
            Number of templates deleted
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM templates WHERE person_id = ?', (person_id,))
            deleted = cursor.rowcount
            self.conn.commit()

            self._log_audit('DELETE_PERSON', None, person_id, f"Deleted {deleted} templates")
            return deleted

        except Exception as e:
            print(f"Error deleting person templates: {e}")
            return 0

    def _update_last_accessed(self, fingerprint_id: str):
        """Update the last_accessed timestamp for a template."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE templates SET last_accessed = CURRENT_TIMESTAMP
                WHERE fingerprint_id = ?
            ''', (fingerprint_id,))
            self.conn.commit()
        except:
            pass  # Non-critical, ignore errors

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with counts, pattern distribution, etc.
        """
        try:
            cursor = self.conn.cursor()

            # Total templates
            cursor.execute('SELECT COUNT(*) as total FROM templates')
            total = cursor.fetchone()['total']

            # Unique persons
            cursor.execute('SELECT COUNT(DISTINCT person_id) as persons FROM templates')
            persons = cursor.fetchone()['persons']

            # Pattern type distribution
            cursor.execute('''
                SELECT pattern_type, COUNT(*) as count 
                FROM templates 
                GROUP BY pattern_type
            ''')
            pattern_dist = {row['pattern_type']: row['count'] for row in cursor.fetchall()}

            # Average minutiae count
            cursor.execute('SELECT AVG(minutiae_count) as avg_minutiae FROM templates')
            avg_minutiae = cursor.fetchone()['avg_minutiae'] or 0

            return {
                'total_templates': total,
                'unique_persons': persons,
                'pattern_distribution': pattern_dist,
                'avg_minutiae_count': round(avg_minutiae, 1)
            }

        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        """Support context manager (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context."""
        self.close()