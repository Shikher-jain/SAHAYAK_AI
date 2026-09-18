import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Document, Page, pdfjs } from 'react-pdf';
// Set up standard worker for react-pdf
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

export default function CitationViewer({ documentUrl, pageNumber, bbox }) {
  const [pageDetails, setPageDetails] = useState(null);

  const onRenderSuccess = (page) => {
    // Capture the original PDF internal dimensions (pt)
    const [originalWidth, originalHeight] = page.originalDocumentScale 
        ? [page.originalWidth, page.originalHeight]
        : page.getViewport({ scale: 1 }).viewBox.slice(2);
    
    // Capture rendered dimensions in the DOM (px)
    setPageDetails({
      originalWidth,
      originalHeight,
      renderedWidth: page.width,
      renderedHeight: page.height
    });
  };

  const getHighlightStyle = () => {
    if (!pageDetails || !bbox || bbox.length !== 4) return { display: 'none' };
    
    const [x0, y0, x1, y1] = bbox;
    const { originalWidth, originalHeight, renderedWidth, renderedHeight } = pageDetails;
    
    const scaleX = renderedWidth / originalWidth;
    const scaleY = renderedHeight / originalHeight;

    return {
      position: 'absolute',
      left: `${x0 * scaleX}px`,
      top: `${y0 * scaleY}px`,
      width: `${(x1 - x0) * scaleX}px`,
      height: `${(y1 - y0) * scaleY}px`,
    };
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto bg-slate-900 border border-slate-800 rounded shadow-xl overflow-hidden">
      <div className="p-3 bg-slate-800 border-b border-slate-700">
        <h3 className="text-slate-200 font-semibold text-sm">Document Citation (Page {pageNumber})</h3>
      </div>
      
      <div className="relative flex justify-center p-4 bg-slate-950 overflow-auto">
        <Document
          file={documentUrl}
          loading={<div className="text-slate-400 p-8">Loading PDF...</div>}
          error={<div className="text-red-400 p-8">Failed to load PDF.</div>}
        >
          <Page 
            pageNumber={pageNumber} 
            renderTextLayer={true}
            renderAnnotationLayer={false}
            onRenderSuccess={onRenderSuccess}
            className="shadow-md relative"
          >
             {/* Bounding box overlay */}
             {pageDetails && bbox && (
               <div 
                 style={getHighlightStyle()} 
                 className="bg-cyan-500/25 border border-cyan-400 rounded-sm pointer-events-none mix-blend-multiply z-10"
               />
             )}
          </Page>
        </Document>
      </div>
    </div>
  );
}

CitationViewer.propTypes = {
  documentUrl: PropTypes.string.isRequired,
  pageNumber: PropTypes.number.isRequired,
  bbox: PropTypes.arrayOf(PropTypes.number)
};
