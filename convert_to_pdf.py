"""
Convert Research Paper Markdown to PDF
Simple method using markdown2 and pdfkit
"""
import os

# Method 1: Using online converter (easiest)
print("="*60)
print("METHOD 1: Online Converter (Recommended)")
print("="*60)
print("\n1. Go to: https://www.markdowntopdf.com/")
print("2. Upload: Research_Paper_With_Figures.md")
print("3. Click 'Convert'")
print("4. Download the PDF")
print("\nOR")
print("\n1. Go to: https://cloudconvert.com/md-to-pdf")
print("2. Upload your markdown file")
print("3. Convert and download")

# Method 2: Using VS Code (if installed)
print("\n" + "="*60)
print("METHOD 2: Using VS Code")
print("="*60)
print("\n1. Open Research_Paper_With_Figures.md in VS Code")
print("2. Press Ctrl+Shift+P")
print("3. Type 'Markdown: Export to PDF'")
print("4. Press Enter")
print("\nNote: Requires 'Markdown PDF' extension")

# Method 3: Using Word
print("\n" + "="*60)
print("METHOD 3: Using Microsoft Word")
print("="*60)
print("\n1. Open Research_Paper_With_Figures.md in VS Code or Notepad")
print("2. Copy all content (Ctrl+A, Ctrl+C)")
print("3. Open Microsoft Word")
print("4. Paste (Ctrl+V)")
print("5. File → Save As → PDF")

# Method 4: Using Pandoc (if installed)
print("\n" + "="*60)
print("METHOD 4: Using Pandoc (Command Line)")
print("="*60)
print("\nIf you have Pandoc installed:")
print("pandoc Research_Paper_With_Figures.md -o Research_Paper.pdf")
print("\nTo install Pandoc:")
print("Download from: https://pandoc.org/installing.html")

print("\n" + "="*60)
print("EASIEST: Use Method 1 (online converter)")
print("="*60)
