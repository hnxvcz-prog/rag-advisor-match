import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.document_parser import DocumentParser

def main():
    print("Starting re-processing of all advisor documents...")
    parser = DocumentParser()
    docs = parser.process_all()
    print(f"Successfully re-processed {len(docs)} documents.")

if __name__ == "__main__":
    main()
