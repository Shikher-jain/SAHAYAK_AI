import React, { useState, useEffect } from 'react';
import { 
  Book, Download, Search, BookOpen, RefreshCw 
} from 'lucide-react';

import { useAppContext } from '../context/AppContext';
import { callBackend } from '../api/client';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { PageHeader } from '../components/ui/PageHeader';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { EmptyState } from '../components/ui/EmptyState';
import { ErrorState } from '../components/ui/ErrorState';
import { Modal } from '../components/ui/Modal';

export const Books = () => {
  const { setCurrentPage } = useAppContext();
  const [books, setBooks] = useState([]);

  const [subjects, setSubjects] = useState([]);
  const [classes, setClasses] = useState([]);
  const [selectedSubject, setSelectedSubject] = useState('all');
  const [selectedClass, setSelectedClass] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedBook, setSelectedBook] = useState(null);

  const fetchCatalog = async () => {
    setLoading(true);
    setError(null);

    const [catalogRes, subjectsRes, classesRes] = await Promise.all([
      callBackend('get', '/books/catalog'),
      callBackend('get', '/books/meta/subjects'),
      callBackend('get', '/books/meta/classes')
    ]);

    if (catalogRes.ok && catalogRes.data) {
      const items = Array.isArray(catalogRes.data) ? catalogRes.data : (catalogRes.data.books || []);
      setBooks(items);
    } else {
      setError(catalogRes.error);
      // Fallback sample catalog if backend returns empty or in dev mode
      setBooks([
        {
          id: 'ncert-math-10',
          title: 'Mathematics (Class X)',
          subject: 'Mathematics',
          class_level: 10,
          description: 'Standard NCERT Mathematics textbook covering Quadratic Equations, Trigonometry, Statistics, and Coordinate Geometry.',
          chapters: ['Real Numbers', 'Polynomials', 'Quadratic Equations', 'Arithmetic Progressions', 'Triangles', 'Trigonometry'],
          url: 'https://ncert.nic.in/textbook.php?jemh1=0-15'
        },
        {
          id: 'ncert-science-10',
          title: 'Science & Technology (Class X)',
          subject: 'Science',
          class_level: 10,
          description: 'Comprehensive physics, chemistry, and biology textbook focusing on Life Processes, Optics, Chemical Reactions, and Electricity.',
          chapters: ['Chemical Reactions', 'Acids, Bases & Salts', 'Metals & Non-metals', 'Life Processes', 'Control & Coordination', 'Light Reflection'],
          url: 'https://ncert.nic.in/textbook.php?jesc1=0-16'
        },
        {
          id: 'ncert-physics-12',
          title: 'Physics Part I & II (Class XII)',
          subject: 'Physics',
          class_level: 12,
          description: 'Senior secondary physics textbook covering Electrostatics, Current Electricity, Magnetism, Optics, and Modern Physics.',
          chapters: ['Electric Charges & Fields', 'Electrostatic Potential', 'Current Electricity', 'Moving Charges & Magnetism', 'Electromagnetic Induction'],
          url: 'https://ncert.nic.in/textbook.php?leph1=0-8'
        },
        {
          id: 'ncert-chem-12',
          title: 'Chemistry (Class XII)',
          subject: 'Chemistry',
          class_level: 12,
          description: 'NCERT textbook covering Physical, Inorganic, and Organic Chemistry with step-by-step problem sets.',
          chapters: ['Solutions', 'Electrochemistry', 'Chemical Kinetics', 'd- and f-Block Elements', 'Coordination Compounds', 'Haloalkanes'],
          url: 'https://ncert.nic.in/textbook.php?lech1=0-9'
        }
      ]);
    }

    if (subjectsRes.ok && subjectsRes.data) {
      setSubjects(Array.isArray(subjectsRes.data) ? subjectsRes.data : []);
    }
    if (classesRes.ok && classesRes.data) {
      setClasses(Array.isArray(classesRes.data) ? classesRes.data : []);
    }

    setLoading(false);
  };

  useEffect(() => {
    fetchCatalog();
  }, []);

  const filteredBooks = books.filter((b) => {
    const matchesSearch = b.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      b.subject?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (b.chapters && b.chapters.some(c => c.toLowerCase().includes(searchQuery.toLowerCase())));
    const matchesSubject = selectedSubject === 'all' || b.subject?.toLowerCase() === selectedSubject.toLowerCase();
    const matchesClass = selectedClass === 'all' || String(b.class_level) === String(selectedClass);
    return matchesSearch && matchesSubject && matchesClass;
  });

  const allSubjects = subjects.length > 0 
    ? subjects 
    : Array.from(new Set(books.map(b => b.subject).filter(Boolean)));

  const allClasses = classes.length > 0 
    ? classes 
    : Array.from(new Set(books.map(b => b.class_level).filter(Boolean))).sort((a,b) => a-b);

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in text-left">
      <PageHeader
        title="Books & NCERT Catalog"
        subtitle="Access curated, open educational resources, NCERT textbooks, and syllabus chapters for self-paced study."
        badge={<Badge variant="primary" size="md">Library</Badge>}
        action={
          <Button
            size="sm"
            variant="ghost"
            icon={RefreshCw}
            loading={loading}
            onClick={fetchCatalog}
          >
            Refresh
          </Button>
        }
      />

      {/* Filters & Search */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-end">
        <div className="sm:col-span-6">
          <Input
            label="Search Catalog"
            placeholder="Search by book title, subject, or chapter..."
            icon={Search}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="sm:col-span-3">
          <Select
            label="Filter Subject"
            value={selectedSubject}
            onChange={(e) => setSelectedSubject(e.target.value)}
          >
            <option value="all">All Subjects</option>
            {allSubjects.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </Select>
        </div>
        <div className="sm:col-span-3">
          <Select
            label="Class / Grade"
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
          >
            <option value="all">All Classes</option>
            {allClasses.map((c) => (
              <option key={c} value={c}>Class {c}</option>
            ))}
          </Select>
        </div>
      </div>

      {error && (
        <ErrorState
          title="Could not connect to books catalog"
          error={error}
          onRetry={fetchCatalog}
        />
      )}

      {/* Loading Skeletons */}
      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Skeleton variant="card" count={4} />
        </div>
      )}

      {/* Empty State */}
      {!loading && filteredBooks.length === 0 && (
        <EmptyState
          icon={Book}
          title="No Books Found"
          description="Try changing the subject or class filter, or modify your search keyword."
          actionLabel="Reset Filters"
          onAction={() => { setSearchQuery(''); setSelectedSubject('all'); setSelectedClass('all'); }}
        />
      )}

      {/* Books Grid */}
      {!loading && filteredBooks.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {filteredBooks.map((book) => (
            <Card
              key={book.id || book.title}
              className="flex flex-col justify-between p-6 space-y-4 hover:border-indigo-300 dark:hover:border-indigo-700 transition-all shadow-sm"
            >
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="p-2.5 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 shrink-0">
                    <Book size={22} />
                  </div>
                  <div className="flex gap-1.5">
                    {book.class_level && (
                      <Badge variant="neutral" size="sm">Class {book.class_level}</Badge>
                    )}
                    {book.subject && (
                      <Badge variant="primary" size="sm">{book.subject}</Badge>
                    )}
                  </div>
                </div>

                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    {book.title}
                  </h3>
                  {book.description && (
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed line-clamp-2">
                      {book.description}
                    </p>
                  )}
                </div>

                {/* Chapters pill list */}
                {book.chapters && book.chapters.length > 0 && (
                  <div className="space-y-1.5 pt-1">
                    <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                      Chapters ({book.chapters.length}):
                    </p>
                    <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto p-1 bg-slate-50 dark:bg-slate-950/50 rounded-xl border border-slate-200/60 dark:border-slate-800/60">
                      {book.chapters.map((ch, idx) => (
                        <span
                          key={idx}
                          className="text-[11px] px-2 py-0.5 rounded-md bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200/60 dark:border-slate-700/60"
                        >
                          {idx + 1}. {ch}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="pt-4 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  icon={BookOpen}
                  onClick={() => setSelectedBook(book)}
                >
                  Chapter Details
                </Button>

                {book.url && (
                  <a
                    href={book.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline"
                  >
                    <span>Download PDF</span>
                    <Download size={14} />
                  </a>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Book Detail Modal */}
      {selectedBook && (
        <Modal
          isOpen={Boolean(selectedBook)}
          onClose={() => setSelectedBook(null)}
          title={selectedBook.title}
          subtitle={`${selectedBook.subject} • Class ${selectedBook.class_level || 'N/A'}`}
          actions={
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSelectedBook(null)}
              >
                Close
              </Button>
              <Button
                size="sm"
                icon={BookOpen}
                onClick={() => {
                  setSelectedBook(null);
                  setCurrentPage('search');
                }}
              >
                Ask AI About This Book
              </Button>
            </>
          }
        >
          <div className="space-y-4 text-xs text-slate-700 dark:text-slate-300 text-left">
            <p className="leading-relaxed">{selectedBook.description}</p>
            
            {selectedBook.chapters && (
              <div>
                <h4 className="font-bold text-slate-900 dark:text-white mb-2">Full Table of Contents:</h4>
                <ol className="space-y-1.5 pl-4 list-decimal">
                  {selectedBook.chapters.map((c, i) => (
                    <li key={i} className="font-medium">{c}</li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
};
