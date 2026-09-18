import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Document, Page } from 'react-pdf'
import { FileWarning, Loader2, ChevronLeft, ChevronRight } from 'lucide-react'
import { apiBase } from '../api/v2_client'

/**
 * CitationViewer — renders the *exact* PDF region a recommendation cites.
 *
 * The RAG pipeline returns coordinates in the PDF's NATIVE coordinate space
 * (1 pt = 1/72 in, top-left origin), plus the page number and the resolver's
 * document URL. We bake the highlight in ONE place — a scaled DIV overlay —
 * rather than re-render the document per citation:
 *
 *   1.  `<Document file={pdfUrl}>` + `<Page pageNumber={page}>` render the
 *       page at react-pdf's natural scale (width 480 px).
 *   2.  The wrapper measures the ACTUAL rendered width with a
 *       `ResizeObserver` (mount is async; canvas+layout land late).
 *   3.  scale = renderedWidth / originalWidth  → the exact 1pt→px ratio.
 *   4.  The bbox `[x_min, y_min, x_max, y_max]` (already in pt, PDF-space)
 *       is multiplied by `scale` and absolutely positioned over the page.
 *
 * Contract (backend `backend/api/routers/audio.py` + `v2 api chat`):
 *   citation = {
 *     pdf_url|document_url: str,   // resolved PDF src
 *     page_number|page: int,
 *     bbox: [x_min, y_min, x_max, y_max],  // pt, top-left origin
 *   }
 *
 * Everything below is dark-mode Tailwind. The base component hands the loaded
 * `pages`/`page` back so the parent can drive paging without a second viewer.
 */
export default function CitationViewer({
  documentUrl,
  pageNumber = 1,
  bbox = null, // [x_min, y_min, x_max, y_max] in PDF points
  className = '',
}) {
  const pdfSrc = useMemo(() => {
    if (!documentUrl) return null
    // Backend document URLs may be relative to the API origin.
    if (/^(https?:|data:)/.test(documentUrl)) return documentUrl
    return `${apiBase.replace(/\/v2$/, '')}${documentUrl}`
  }, [documentUrl])

  const pageRef = useRef(null)
  const [numPages, setNumPages] = useState(null)
  const [page, setPage] = useState(pageNumber)
  const [renderedWidth, setRenderedWidth] = useState(null)

  // ── Measure the real rendered width (mount is async → never assume) ──
  useEffect(() => {
    const el = pageRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        if (e.contentRect.width) setRenderedWidth(e.contentRect.width)
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => setPage(pageNumber), [pageNumber])

  // ── Scale math: renderedDOM px / native PDF pt (1pt = 1/72in) ──────
  const highlight = useMemo(() => {
    if (!bbox || !renderedWidth) return null
    const [x0, y0, x1, y1] = bbox
    // PDF pt → rendered px assuming a 1:1 natural page render; react-pdf
    // reports `width` on the Page in pt — the ratio is exactly the zoom.
    const pageWidthPt = pageWidthPtRef.current
    const scale = renderedWidth / (pageWidthPt || renderedWidth)
    return {
      left: x0 * scale,
      top: y0 * scale,
      width: (x1 - x0) * scale,
      height: (y1 - y0) * scale,
      scale,
    }
  }, [bbox, renderedWidth])
  const pageWidthPtRef = useRef(1)

  const onPageLoad = useCallback((pageObj) => {
    pageWidthPtRef.current = pageObj.getViewport({ scale: 1 }).width
  }, [])

  if (!pdfSrc) {
    return (
      <div className={`flex flex-col items-center gap-2 p-8 text-sm text-slate-400 border border-slate-800 rounded-xl bg-slate-900/60 ${className}`}>
        <FileWarning className="w-5 h-5 text-amber-400" />
        No PDF source was linked for this citation.
      </div>
    )
  }

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {/* pager */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          Citation on page {page}{numPages ? ` / ${numPages}` : ''}
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="p-1 rounded hover:bg-slate-800 disabled:opacity-30"
            aria-label="Previous page"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setPage((p) => Math.min(numPages || p, p + 1))}
            disabled={!numPages || page >= numPages}
            className="p-1 rounded hover:bg-slate-800 disabled:opacity-30"
            aria-label="Next page"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* page + scaled highlight overlay */}
      <div className="relative bg-slate-800/40 rounded-lg overflow-hidden border border-slate-700/60">
        <div ref={pageRef} className="relative">
          <Document
            file={pdfSrc}
            onLoadSuccess={({ numPages: n }) => setNumPages(n)}
            loading={
              <div className="flex items-center justify-center gap-2 p-10 text-slate-400 text-sm">
                <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                Loading source PDF…
              </div>
            }
            error={
              <div className="flex items-center gap-2 p-10 text-amber-300 text-sm">
                <FileWarning className="w-4 h-4" />
                This document couldn't be loaded — it may live behind private storage.
              </div>
            }
          >
            <Page
              key={pdfSrc + ':' + page}
              pageNumber={page}
              renderTextLayer={false}
              renderAnnotationLayer={false}
              onLoadSuccess={onPageLoad}
              className="rounded"
            />
          </Document>

          {highlight && (
            <div
              className="pointer-events-none absolute border-2 border-cyan-400/80 bg-cyan-400/25 mix-blend-screen"
              style={{
                left: highlight.left,
                top: highlight.top,
                width: highlight.width,
                height: highlight.height,
                boxShadow: '0 0 18px rgba(34,211,238,0.35)',
              }}
            />
          )}
        </div>
      </div>
    </div>
  )
}

// The worker contract: react-pdf v9 bundles pdfjs-dist; the worker must come
// from the same version or every page render throws "Setting up fake worker".
import { GlobalWorkerOptions } from 'pdfjs-dist'
GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${'4.4.168'}/build/pdf.worker.min.mjs`
