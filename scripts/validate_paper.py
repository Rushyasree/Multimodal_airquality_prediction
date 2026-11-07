import re, glob, os

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    main_tex = f.read()

# Check included tables
tables = re.findall(r'\\input\{([^}]+)\}', main_tex)
print(f'Tables found: {len(tables)}')
for t in tables:
    p = os.path.join('paper', t if t.endswith('.tex') else t + '.tex')
    exists = os.path.exists(p)
    print(f'  Table {t}: exists={exists}')
    if exists:
        with open(p, 'r', encoding='utf-8') as tf:
            content = tf.read()
            if content.count('{') != content.count('}'):
                print(f'    WARNING: unbalanced braces in {t} ({content.count("{")} vs {content.count("}")})')

# Check figures
figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', main_tex)
print(f'Figures found: {len(figures)}')
for fig in figures:
    p = os.path.join('paper', fig)
    p_fig = os.path.join('paper', 'figures', os.path.basename(fig))
    exists = os.path.exists(p) or os.path.exists(p_fig)
    print(f'  Figure {fig}: exists={exists}')

# Check citations
cites = set(re.findall(r'\\cite\{([^}]+)\}', main_tex))
all_cites = set()
for c in cites:
    for item in c.split(','):
        all_cites.add(item.strip())
with open('paper/references.bib', 'r', encoding='utf-8') as f:
    bib_text = f.read()
bib_keys = set(re.findall(r'@[a-zA-Z]+\{([^,]+),', bib_text))
print(f'Citations used: {len(all_cites)}')
print(f'Bib keys available: {len(bib_keys)}')
missing_bib = all_cites - bib_keys
if missing_bib:
    print(f'  WARNING: Missing bib keys: {missing_bib}')
else:
    print('  All citations match references.bib successfully!')

# Check main.tex braces
if main_tex.count('{') != main_tex.count('}'):
    print(f'WARNING: unbalanced braces in main.tex ({main_tex.count("{")} vs {main_tex.count("}")})')
else:
    print('main.tex braces perfectly balanced!')
