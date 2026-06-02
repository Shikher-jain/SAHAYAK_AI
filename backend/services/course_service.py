"""Course service — curated external course catalog with search and recommendations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Pre-loaded catalog of best free courses by subject
_COURSE_CATALOG: List[Dict[str, Any]] = [
    # --- Data Science ---
    {"id": "ds-1", "title": "Data Science Specialization", "provider": "Coursera (Johns Hopkins)", "subject": "Data Science", "level": "Beginner", "url": "https://www.coursera.org/specializations/data-science-foundations", "price": "Free (audit)", "duration": "10 months", "rating": 4.6},
    {"id": "ds-2", "title": "Applied Data Science with Python", "provider": "Coursera (Michigan)", "subject": "Data Science", "level": "Intermediate", "url": "https://www.coursera.org/specializations/data-science-python", "price": "Free (audit)", "duration": "5 months", "rating": 4.4},
    {"id": "ds-3", "title": "Data Science Fundamentals", "provider": "Khan Academy", "subject": "Data Science", "level": "Beginner", "url": "https://www.khanacademy.org/math/statistics-probability", "price": "Free", "duration": "Self-paced", "rating": 4.5},
    # --- Machine Learning ---
    {"id": "ml-1", "title": "Machine Learning Specialization", "provider": "Coursera (Andrew Ng)", "subject": "Machine Learning", "level": "Beginner", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "price": "Free (audit)", "duration": "3 months", "rating": 4.9},
    {"id": "ml-2", "title": "Deep Learning Specialization", "provider": "Coursera (deeplearning.ai)", "subject": "Deep Learning", "level": "Intermediate", "url": "https://www.coursera.org/specializations/deep-learning", "price": "Free (audit)", "duration": "5 months", "rating": 4.8},
    {"id": "ml-3", "title": "CS229: Machine Learning", "provider": "Stanford Online", "subject": "Machine Learning", "level": "Advanced", "url": "https://online.stanford.edu/courses/cs229-machine-learning", "price": "Free", "duration": "11 weeks", "rating": 4.7},
    # --- Web Development ---
    {"id": "web-1", "title": "Full-Stack Web Development", "provider": "freeCodeCamp", "subject": "Web Development", "level": "Beginner", "url": "https://www.freecodecamp.org/learn", "price": "Free", "duration": "Self-paced", "rating": 4.8},
    {"id": "web-2", "title": "CS50's Web Programming", "provider": "Harvard (edX)", "subject": "Web Development", "level": "Intermediate", "url": "https://pll.harvard.edu/course/cs50s-web-programming-python-and-javascript", "price": "Free", "duration": "12 weeks", "rating": 4.7},
    # --- DSA ---
    {"id": "dsa-1", "title": "Data Structures and Algorithms", "provider": "Khan Academy", "subject": "DSA", "level": "Beginner", "url": "https://www.khanacademy.org/computing/computer-science/algorithms", "price": "Free", "duration": "Self-paced", "rating": 4.6},
    {"id": "dsa-2", "title": "Algorithms Specialization", "provider": "Coursera (Stanford)", "subject": "DSA", "level": "Intermediate", "url": "https://www.coursera.org/specializations/algorithms", "price": "Free (audit)", "duration": "4 months", "rating": 4.7},
    # --- Mathematics ---
    {"id": "math-1", "title": "Linear Algebra", "provider": "Khan Academy", "subject": "Mathematics", "level": "Beginner", "url": "https://www.khanacademy.org/math/linear-algebra", "price": "Free", "duration": "Self-paced", "rating": 4.5},
    {"id": "math-2", "title": "Mathematics for ML", "provider": "Coursera (Imperial College)", "subject": "Mathematics", "level": "Intermediate", "url": "https://www.coursera.org/specializations/mathematics-machine-learning", "price": "Free (audit)", "duration": "3 months", "rating": 4.5},
    # --- Programming ---
    {"id": "prog-1", "title": "Python for Everybody", "provider": "Coursera (Michigan)", "subject": "Programming", "level": "Beginner", "url": "https://www.coursera.org/specializations/python", "price": "Free (audit)", "duration": "8 months", "rating": 4.8},
    {"id": "prog-2", "title": "Introduction to Computer Science", "provider": "Harvard (edX)", "subject": "Programming", "level": "Beginner", "url": "https://pll.harvard.edu/course/cs50-introduction-computer-science", "price": "Free", "duration": "12 weeks", "rating": 4.9},
]


def get_course_count() -> int:
    return len(_COURSE_CATALOG)


def list_courses(subject: Optional[str] = None, level: Optional[str] = None) -> List[Dict[str, Any]]:
    """List courses with optional filters."""
    results = _COURSE_CATALOG
    if subject:
        results = [c for c in results if c["subject"].lower() == subject.lower()]
    if level:
        results = [c for c in results if c["level"].lower() == level.lower()]
    return results


def get_course(course_id: str) -> Optional[Dict[str, Any]]:
    for course in _COURSE_CATALOG:
        if course["id"] == course_id:
            return course
    return None


def add_custom_source(course: Dict[str, Any]) -> Dict[str, Any]:
    """Add a user-provided course/source to the catalog."""
    new_id = f"custom-{len(_COURSE_CATALOG) + 1}"
    entry = {"id": new_id, **course}
    _COURSE_CATALOG.append(entry)
    return entry


def search_courses(query: str) -> List[Dict[str, Any]]:
    """Simple keyword search across course titles, subjects, and providers."""
    q = query.lower()
    return [
        c for c in _COURSE_CATALOG
        if q in c["title"].lower() or q in c["subject"].lower() or q in c["provider"].lower()
    ]
