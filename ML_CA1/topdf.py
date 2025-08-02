import markdown
import sys
import os
from weasyprint import HTML, CSS
from pathlib import Path
import logging

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_markdown_to_pdf(markdown_file_path, output_pdf_path=None, custom_css_path=None):
    """
    Converts a Markdown file to a PDF file with proper formatting, syntax highlighting,
    tables, and image handling.
    """
    if not os.path.exists(markdown_file_path):
        logging.error(f"Error: Markdown file not found at '{markdown_file_path}'")
        sys.exit(1)

    markdown_file_path = Path(markdown_file_path)
    if output_pdf_path is None:
        output_pdf_path = markdown_file_path.with_suffix('.pdf')
    else:
        output_pdf_path = Path(output_pdf_path)

    md_extensions = [
        'extra',
        'codehilite',
        'fenced_code',
    ]

    # A base CSS for basic styling.
    default_css = CSS(string="""
        @page { size: A4; margin: 2cm; }
        body { font-family: sans-serif; line-height: 1.6; }
        h1, h2, h3, h4, h5, h6 { margin-top: 1em; margin-bottom: 0.5em; }
        pre { background-color: #f4f4f4; padding: 1em; border-radius: 5px; }
        img { max-width: 100%; height: auto; }
        table { border-collapse: collapse; width: 100%; margin: 1em 0; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    """)
    
    pygments_css = CSS(url='https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/default.min.css')

    try:
        logging.info(f"Reading Markdown file from: {markdown_file_path}")
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        logging.info("Converting Markdown to HTML...")
        html_content = markdown.markdown(
            markdown_text,
            extensions=md_extensions,
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'linenos': False,
                    'pygments_style': 'default'
                }
            }
        )

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{markdown_file_path.stem}</title>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # CORRECTED: Resolve the parent path to an absolute path before converting to a URI
        base_url = markdown_file_path.parent.resolve().as_uri() + '/'
        
        logging.info("Rendering HTML to PDF with WeasyPrint...")
        html = HTML(string=full_html, base_url=base_url)

        stylesheets = [default_css, pygments_css]
        if custom_css_path and os.path.exists(custom_css_path):
            logging.info(f"Applying custom CSS from: {custom_css_path}")
            stylesheets.append(CSS(filename=custom_css_path))

        html.write_pdf(output_pdf_path, stylesheets=stylesheets)

        logging.info(f"Successfully converted '{markdown_file_path}' to '{output_pdf_path}'")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python md_to_pdf.py <input_markdown_file> [<output_pdf_file>] [--css <custom_css_file>]")
        print("Example: python md_to_pdf.py my_document.md")
        sys.exit(0)

    input_file = sys.argv[1]
    output_file = None
    css_file = None

    if len(sys.argv) > 2:
        for i, arg in enumerate(sys.argv[2:]):
            if arg == '--css' and (i + 3) <= len(sys.argv):
                css_file = sys.argv[i + 3]
            elif not arg.startswith('--'):
                output_file = arg
    
    convert_markdown_to_pdf(input_file, output_file, css_file)