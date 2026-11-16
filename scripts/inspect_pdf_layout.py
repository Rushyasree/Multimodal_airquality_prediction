import fitz
from pathlib import Path

doc = fitz.open('paper/VayuDrishti_IEEE_Manuscript.pdf')
print(f"Total pages: {len(doc)}")

for pno, page in enumerate(doc):
    print(f"\n================ PAGE {pno+1} ================")
    rect = page.rect
    print(f"Page size: {rect.width} x {rect.height} pt ({rect.width/72:.2f} x {rect.height/72:.2f} in)")
    
    # Check text blocks
    blocks = page.get_text("blocks")
    print(f"Text blocks count: {len(blocks)}")
    
    # Check images
    img_list = page.get_images()
    print(f"Images count: {len(img_list)}")
    
    # Check drawings (lines, borders)
    drawings = page.get_drawings()
    print(f"Vector drawings count: {len(drawings)}")
    
    for bno, b in enumerate(blocks[:6]):
        x0, y0, x1, y1, text, block_no, block_type = b
        first_line = text.strip().split('\n')[0] if text.strip() else "[empty]"
        print(f"  Block {bno}: y=[{y0:.1f}, {y1:.1f}], x=[{x0:.1f}, {x1:.1f}] -> {first_line[:60]}")
