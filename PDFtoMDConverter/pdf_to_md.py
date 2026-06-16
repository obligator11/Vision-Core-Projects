import sys
import os
from pathlib import Path

try:
    import pymupdf4llm
except ImportError:
    print("\n[!] Error: 'pymupdf4llm' package not found.")
    print("[*] Resolution: Run 'pip install pymupdf4llm' to install it offline.\n")
    sys.exit(1)


class AutomatedPDFPipeline:
    """Manages an automated local folder pipeline for processing PDFs into Markdown."""

    def __init__(self, raw_dir: str = "raw_pdfs", output_dir: str = "markdown_output"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        
        # Self-create workspace folders automatically if missing
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_pipeline(self):
        """Scans folder, skips converted documents, parses new items, and cleans space."""
        # Find all PDF items sitting inside the raw folder
        pdf_files = list(self.raw_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"[*] No files found. Drop new PDF documents directly inside: /{self.raw_dir}")
            return

        print(f"[*] Found {len(pdf_files)} items in folder stream. Checking sync...")

        for pdf_path in pdf_files:
            # Establish the matching output target destination name
            output_filename = self.output_dir / f"{pdf_path.stem}.md"

            # Check if this document has already been processed before
            if output_filename.exists():
                print(f"[=] Skipping: '{pdf_path.name}' (Matching Markdown already exists!)")
                continue

            print(f"[*] Converting raw binary layer: {pdf_path.name}")
            
            try:
                # Local matrix conversion process execution loop
                markdown_content = pymupdf4llm.to_markdown(str(pdf_path))
                
                # Check for empty files safely
                if not markdown_content.strip():
                    print(f"[!] Warning: Text layer on '{pdf_path.name}' came back blank.")

                # Commit text layout to disk via binary safe UTF-8 streams
                output_filename.write_bytes(markdown_content.encode("utf-8"))
                print(f"[+] Saved clean layout to: {output_filename}")

                # Auto-delete original file to preserve local disk space
                pdf_path.unlink()
                print(f"[-] Purged original source item: {pdf_path.name}")

            except Exception as file_error:
                print(f"[-] Failed processing asset '{pdf_path.name}': {str(file_error)}")


if __name__ == "__main__":
    print("=" * 60)
    print("        AUTOMATED OFFLINE TOKEN-OPTIMIZER PIPELINE      ")
    print("=" * 60)

    try:
        # Initialize pipeline infrastructure folders automatically
        pipeline = AutomatedPDFPipeline(raw_dir="raw_pdfs", output_dir="markdown_output")
        pipeline.process_pipeline()
    except Exception as error:
        print(f"\n[-] Critical System Exception encountered: {str(error)}")