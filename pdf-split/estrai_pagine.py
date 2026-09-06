from pathlib import Path
from pypdf import PdfReader, PdfWriter

# ── Configurazione ──────────────────────────────────────────
INPUT_PDF = r"C:\Users\massi\Desktop\addlance - ssm\testo-completo.pdf"  # <-- cambia
OUTPUT_PDF = r"capitolo3.pdf"
PAGINA_DA = 208  # prima pagina da estrarre (numerazione FISICA del viewer, 1-based)
PAGINA_A = 271  # ultima pagina da estrarre (inclusa)
# ────────────────────────────────────────────────────────────

reader = PdfReader(INPUT_PDF)
tot = len(reader.pages)
print(f"PDF di input: {tot} pagine")

if not (1 <= PAGINA_DA <= PAGINA_A <= tot):
    raise SystemExit(f"Range non valido: {PAGINA_DA}-{PAGINA_A} (il PDF ha {tot} pagine)")

writer = PdfWriter()
for i in range(PAGINA_DA - 1, PAGINA_A):  # da 1-based a 0-based
    writer.add_page(reader.pages[i])

out = Path(OUTPUT_PDF)
with out.open("wb") as f:
    writer.write(f)

print(f"Scritto {out.resolve()} con {PAGINA_A - PAGINA_DA + 1} pagine ({PAGINA_DA}-{PAGINA_A})")
