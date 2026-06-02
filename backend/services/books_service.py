"""Books service — NCERT and open textbook catalog with RAG ingestion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# NCERT Book catalog — class 1-12, major subjects
_NCERT_CATALOG: List[Dict[str, Any]] = [
    # Class 10
    {"id": "ncert-10-math", "title": "Mathematics Class 10", "subject": "Mathematics", "class_level": 10, "url": "https://ncert.nic.in/textbook/pdf/jemh1ps.pdf", "chapters": ["Real Numbers", "Polynomials", "Linear Equations", "Quadratic Equations", "Arithmetic Progressions", "Triangles", "Coordinate Geometry", "Trigonometry", "Circles", "Areas", "Constructions", "Surface Areas", "Statistics", "Probability"]},
    {"id": "ncert-10-sci", "title": "Science Class 10", "subject": "Science", "class_level": 10, "url": "https://ncert.nic.in/textbook/pdf/jesc1ps.pdf", "chapters": ["Chemical Reactions", "Acids Bases and Salts", "Metals and Non-metals", "Carbon Compounds", "Life Processes", "Control and Coordination", "Reproduction", "Heredity", "Light", "Human Eye", "Electricity", "Magnetic Effects", "Sources of Energy", "Environment"]},
    {"id": "ncert-10-eng", "title": "English Class 10 (First Flight)", "subject": "English", "class_level": 10, "url": "https://ncert.nic.in/textbook/pdf/jeff1ps.pdf", "chapters": ["A Letter to God", "Nelson Mandela", "Two Stories about Flying", "From the Diary of Anne Frank", "The Hundred Dresses", "Glimpses of India", "Mijbil the Otter", "Madam Rides the Bus", "The Sermon at Benares", "The Proposal"]},
    # Class 11
    {"id": "ncert-11-math", "title": "Mathematics Class 11", "subject": "Mathematics", "class_level": 11, "url": "https://ncert.nic.in/textbook/pdf/kemh1ps.pdf", "chapters": ["Sets", "Relations and Functions", "Trigonometric Functions", "Complex Numbers", "Linear Inequalities", "Permutations", "Binomial Theorem", "Sequences and Series", "Straight Lines", "Conic Sections", "3D Geometry", "Limits and Derivatives", "Statistics", "Probability"]},
    {"id": "ncert-11-phy", "title": "Physics Class 11", "subject": "Physics", "class_level": 11, "url": "https://ncert.nic.in/textbook/pdf/keph1ps.pdf", "chapters": ["Physical World", "Units and Measurements", "Motion in a Straight Line", "Motion in a Plane", "Laws of Motion", "Work Energy and Power", "Gravitation", "Mechanical Properties", "Thermodynamics", "Kinetic Theory", "Oscillations", "Waves"]},
    {"id": "ncert-11-chem", "title": "Chemistry Class 11", "subject": "Chemistry", "class_level": 11, "url": "https://ncert.nic.in/textbook/pdf/kech1ps.pdf", "chapters": ["Some Basic Concepts", "Structure of Atom", "Classification of Elements", "Chemical Bonding", "Thermodynamics", "Equilibrium", "Redox Reactions", "Organic Chemistry", "Hydrocarbons", "Environmental Chemistry"]},
    # Class 12
    {"id": "ncert-12-math", "title": "Mathematics Class 12", "subject": "Mathematics", "class_level": 12, "url": "https://ncert.nic.in/textbook/pdf/lemh1ps.pdf", "chapters": ["Relations and Functions", "Inverse Trigonometric", "Matrices", "Determinants", "Continuity", "Application of Derivatives", "Integrals", "Differential Equations", "Vector Algebra", "3D Geometry", "Linear Programming", "Probability"]},
    {"id": "ncert-12-phy", "title": "Physics Class 12", "subject": "Physics", "class_level": 12, "url": "https://ncert.nic.in/textbook/pdf/leph1ps.pdf", "chapters": ["Electric Charges", "Electrostatic Potential", "Current Electricity", "Moving Charges", "Magnetism", "Electromagnetic Induction", "Alternating Current", "Electromagnetic Waves", "Ray Optics", "Wave Optics", "Dual Nature", "Atoms", "Nuclei", "Semiconductor"]},
    {"id": "ncert-12-chem", "title": "Chemistry Class 12", "subject": "Chemistry", "class_level": 12, "url": "https://ncert.nic.in/textbook/pdf/lech1ps.pdf", "chapters": ["Solid State", "Solutions", "Electrochemistry", "Chemical Kinetics", "Surface Chemistry", "p-Block Elements", "d and f Block", "Coordination Compounds", "Haloalkanes", "Alcohols", "Aldehydes", "Amines", "Biomolecules", "Polymers"]},
    {"id": "ncert-12-cs", "title": "Computer Science Class 12", "subject": "Computer Science", "class_level": 12, "url": "https://ncert.nic.in/textbook/pdf/lecs1ps.pdf", "chapters": ["Python Revision", "File Handling", "Stack", "Queue", "Sorting", "Boolean Algebra", "Computer Networks", "Database Management", "SQL", "Web Technologies"]},
    # Class 9
    {"id": "ncert-9-math", "title": "Mathematics Class 9", "subject": "Mathematics", "class_level": 9, "url": "https://ncert.nic.in/textbook/pdf/iemh1ps.pdf", "chapters": ["Number Systems", "Polynomials", "Coordinate Geometry", "Linear Equations", "Euclid's Geometry", "Lines and Angles", "Triangles", "Quadrilaterals", "Circles", "Heron's Formula", "Surface Areas", "Statistics", "Probability"]},
    {"id": "ncert-9-sci", "title": "Science Class 9", "subject": "Science", "class_level": 9, "url": "https://ncert.nic.in/textbook/pdf/iesc1ps.pdf", "chapters": ["Matter in Our Surroundings", "Is Matter Pure", "Atoms and Molecules", "Structure of Atom", "Fundamental Unit of Life", "Tissues", "Motion", "Force and Laws", "Gravitation", "Work and Energy", "Sound", "Natural Resources"]},
]


def list_books(subject: Optional[str] = None, class_level: Optional[int] = None) -> List[Dict[str, Any]]:
    """List NCERT books with optional filters."""
    results = _NCERT_CATALOG
    if subject:
        results = [b for b in results if b["subject"].lower() == subject.lower()]
    if class_level is not None:
        results = [b for b in results if b["class_level"] == class_level]
    return results


def get_book(book_id: str) -> Optional[Dict[str, Any]]:
    for book in _NCERT_CATALOG:
        if book["id"] == book_id:
            return book
    return None


def get_subjects() -> List[str]:
    return sorted(set(b["subject"] for b in _NCERT_CATALOG))


def get_class_levels() -> List[int]:
    return sorted(set(b["class_level"] for b in _NCERT_CATALOG))
