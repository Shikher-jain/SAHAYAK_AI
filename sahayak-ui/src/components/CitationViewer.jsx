import React, { useState, useRef, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { Loader2, FileText } from 'lucide-react';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Configure the PDF.js web worker (Required for Vite/Next.js)
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export default function CitationViewer({ documentUrl, pageNumber, bbox }) {
  const [numPages, setNumPages] = useState(null);
  const [pageDetails, setPageDetails] = useState(null);
  const [renderedWidth, setRenderedWidth] = useState(0);
  const containerRef = useRef(null);

  // Re-calculate bounding box overlay if window resizes
  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      if (entries[0]) setRenderedWidth(entries[0].contentRect.width);
    });
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const onPageLoadSuccess = (page) => {
    // Get the original, unscaled PDF dimensions (1pt = 1/72 inch)
    const viewport = page.getViewport({ scale: 1 });
    setPageDetails({ originalWidth: viewport.width, originalHeight: viewport.height });
  };

  const renderHighlight = () => {
    if (!bbox || !pageDetails || !renderedWidth) return null;

    const [x_min, y_min, x_max, y_max] = bbox;
    
    // Scale backend coordinates to the current DOM size
    const scale = renderedWidth / pageDetails.originalWidth;

    const left = x_min * scale;
    const top = y_min * scale;
    const width = (x_max - x_min) * scale;
    const height = (y_max - y_min) * scale;

    return (
      <div
        className="absolute bg-cyan-400/30 border-2 border-cyan-400 rounded-sm cursor-pointer transition-all hover:bg-cyan-400/40 shadow-[0_0_15px_rgba(34,211,238,0.4)] pointer-events-none"
        style={{
          left: `${left}px`,
          top: `${top}px`,
          width: `${width}px`,
          height: `${height}px`,
        }}
      />
    );
  };

  return (
    <div className="w-full h-full bg-slate-950 border border-slate-800 rounded-lg overflow-hidden flex flex-col relative">
      <div className="bg-slate-900 border-b border-slate-800 p-3 flex justify-between items-center text-xs text-slate-300">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-cyan-500" />
          <span className="font-semibold text-cyan-400">Source Evidence</span>
        </div>
        <span>Page {pageNumber} {numPages ? `of ${numPages}` : ''}</span>
      </div>
      
      <div className="flex-1 overflow-auto relative p-4 flex justify-center bg-slate-950/50">
        <Document
          file={documentUrl}
          onLoadSuccess={({ numPages }) => setNumPages(numPages)}
          loading={<Loader2 className="w-6 h-6 animate-spin text-cyan-500 mt-10" />}
          className="flex flex-col items-center"
        >
          <div ref={containerRef} className="relative shadow-2xl rounded overflow-hidden border border-slate-700">
            <Page
              pageNumber={pageNumber}
              onLoadSuccess={onPageLoadSuccess}
              width={600} // Base constraint; scales responsively down
              renderTextLayer={true} // Enable text selection
              renderAnnotationLayer={false}
              className="max-w-full"
            />
            {renderHighlight()}
          </div>
        </Document>
      </div>
    </div>
  );
}
