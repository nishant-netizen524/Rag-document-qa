"""PDF text extraction with Page numbers. pdfplumber first, pypdf fallback"""
import logging 
import re

import pdfplumber

logger = logging.getLogger (__name__)

def clean_text(text:str) -> str:
    if not text:
        return ""

    # Wrapped lines inside a paragraph become spaces; paragraph breaks stay
    text = re.sub(r"(?<!\n)\n(?!\n)"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    text = re.sub(r"[\t]+","",text)

    #Drop standalone page-number lines
    text = re.sub(r"^\s*\d+\s*$","",text,flags=re.MULTILINE)
    return text.strip()


def extract_pdf(path: str):
    """Returns (pages,,total_pages). pages = [{"page"}:int, "text":str].

    Raises ValueError with a user-friendly message on bad/encrypted/scanned PDFs.
    """
    try:
        return _extract_with_pdfplumber(path)
    except Exception as exc: #noqa: BLE001 - fallback path
        logger.warning("pdfplumber failed (%s); falling back to pypdf",exc)
        return _extract_with_pypdf(path)

def _extract_with_pdfplumber(path:str):
        
        pdf = pdfplumber.open(path)
        try:
            if pdf.is_encrypted:
                raise ValueError("This PDF is password-protected. Please upload an PDF File")
            pages, total = [],len(pdf.pages)
            for i,page in enumerate(pdf.pages,start=1):
                try:
                    raw = page.extract_text() or ""
                except Exception: # noqa: BLE001 - one bad page shouldn't kill the file
                    raw =""
                text = clean_text(raw)
                if text:
                    pages.append({"page":i,"text":text})

            if not pages:
                raise ValueError("No readable text found. The PDF may be scanned images (OCR is not enabled in v1).")
            return pages,total
        finally:
            pdf.close()

def _extract_with_pypdf(path:str):
    from pypdf import PdfReader

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("This PDF is password-protected. Please upload an unencrypted PDF.")
        
        pages,total = [],len(reader.pages)
        for i,page in enumerate(reader.pages,start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"page":i,"text":text})
        if not pages:
            raise ValueError("No readable text found. The PDF may be scanned images (OCR is not enabled in v1).")
        return pages,total
    except Exception as exc: #noqa: BLE001
        if isinstance(exc,ValueError):
            raise
        raise ValueError("The file is not a valid PDF.") from exc



    