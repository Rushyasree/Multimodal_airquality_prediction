import os
import sys
from pathlib import Path
import win32com.client

ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = ROOT / "docs/VayuDrishti_Final_Capstone_Report_Repaired.docx"
PDF_PATH = ROOT / "docs/VayuDrishti_Final_Capstone_Report_Repaired.pdf"

print(f"Opening Word to update fields and export PDF: {DOCX_PATH.name}...")
word = win32com.client.Dispatch("Word.Application")
word.Visible = False

try:
    doc = word.Documents.Open(str(DOCX_PATH.resolve()))
    
    # Update fields
    print("Updating fields...")
    try:
        doc.Fields.Update()
    except Exception as e:
        print(f"Note on field update: {e}")
        
    for toc in doc.TablesOfContents:
        try:
            toc.Update()
        except Exception as e:
            print(f"Note on TOC update: {e}")
            
    # Save docx
    doc.Save()
    print("Saved updated Word document.")
    
    # Export PDF (FileFormat = 17 -> wdFormatPDF)
    print(f"Exporting PDF to {PDF_PATH.name}...")
    doc.SaveAs2(str(PDF_PATH.resolve()), FileFormat=17)
    
    # Check page count
    pages = doc.ComputeStatistics(2) # 2 = wdStatisticPages
    print(f"Total document pages computed by Word: {pages}")
    
    doc.Close()
finally:
    word.Quit()

print("Word application closed successfully.")
print(f"PDF exists: {PDF_PATH.exists()}, size: {PDF_PATH.stat().st_size} bytes")
