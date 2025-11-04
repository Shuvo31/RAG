import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import hashlib
from rapidocr_onnxruntime import RapidOCR
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directory
DEFAULT_LOCAL_DIR = r"Entity_Resorts"

class SelectiveImageProcessor:
    def __init__(self):
        # Initialize fast OCR engine
        try:
            self.ocr_engine = RapidOCR()
            self.ocr_available = True
            logger.info("✅ Fast OCR engine initialized")
        except Exception as e:
            self.ocr_available = False
            logger.warning(f"❌ OCR not available: {e}")
        
        # Cache setup
        self.cache_dir = Path(".image_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "ocr_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load OCR cache to avoid reprocessing same images"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save OCR cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _get_image_hash(self, image_path):
        """Generate hash for cache key"""
        return hashlib.md5(Path(image_path).read_bytes()).hexdigest()
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using fast OCR with caching"""
        if not self.ocr_available:
            return ""
        
        image_hash = self._get_image_hash(image_path)
        
        # Check cache first
        if image_hash in self.cache:
            logger.debug(f"Using cached result for {image_path.name}")
            return self.cache[image_hash]
        
        try:
            # Perform OCR
            result, _ = self.ocr_engine(str(image_path))
            if result:
                # Combine all detected text
                text = " ".join([res[1] for res in result]).strip()
                
                # Only cache substantial text
                if len(text) > 5:
                    self.cache[image_hash] = text
                    self._save_cache()
                
                return text
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
        
        return ""
    
    def extract_images_from_pdf(self, pdf_path, min_image_size=10000, temp_dir="temp_images"):
        """Extract images from PDF and return OCR results"""
        if not self.ocr_available:
            logger.warning("OCR not available - skipping image extraction")
            return []
        
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        
        extracted_data = []
        
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF: {pdf_path.name} ({len(doc)} pages)")
            
            total_images = 0
            processed_images = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Filter small images (icons, etc.)
                        if pix.width * pix.height < min_image_size:
                            pix = None
                            continue
                        
                        # Save temporary image
                        img_filename = f"{pdf_path.stem}_p{page_num+1}_i{img_index+1}.png"
                        img_path = temp_path / img_filename
                        
                        if pix.n == 4:  # Convert CMYK to RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        pix.save(img_path)
                        total_images += 1
                        
                        # Perform OCR
                        ocr_text = self.extract_text_from_image(img_path)
                        
                        if ocr_text and len(ocr_text) > 10:  # Only keep substantial text
                            extracted_data.append({
                                "pdf_file": pdf_path.name,
                                "pdf_path": str(pdf_path),
                                "page_number": page_num + 1,
                                "image_index": img_index + 1,
                                "image_size": f"{pix.width}x{pix.height}",
                                "ocr_text": ocr_text,
                                "text_length": len(ocr_text),
                                "extracted_at": datetime.now().isoformat()
                            })
                            processed_images += 1
                            logger.info(f"  ✓ Page {page_num+1}: Extracted {len(ocr_text)} chars")
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"  Failed to process image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            logger.info(f"  Processed {processed_images}/{total_images} images from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
        
        # Clean up temporary images
        try:
            for img_file in temp_path.glob("*.png"):
                img_file.unlink()
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
        
        return extracted_data
    
    def process_pdfs(self, input_path=None, output_file="extracted_image_texts.json"):
        """Main method to process PDFs"""
        # Use default directory if no input provided
        if input_path is None:
            input_path = DEFAULT_LOCAL_DIR
        
        input_path = Path(input_path)
        all_extracted_data = []
        
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            pdf_files = [input_path]
        elif input_path.is_dir():
            pdf_files = list(input_path.rglob("*.pdf"))
        else:
            logger.error("Input must be a PDF file or directory containing PDFs")
            return
        
        if not pdf_files:
            logger.error(f"No PDF files found in {input_path}!")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            logger.info(f"🔍 Processing: {pdf_path.name}")
            extracted_data = self.extract_images_from_pdf(pdf_path)
            all_extracted_data.extend(extracted_data)
            logger.info(f"  ✅ Extracted {len(extracted_data)} image texts from {pdf_path.name}")
        
        # Save results
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_extracted_data, f, indent=2, ensure_ascii=False)
        
        # Summary
        total_text_chars = sum(item['text_length'] for item in all_extracted_data)
        logger.info(f"🎉 Processing complete!")
        logger.info(f"📊 Summary:")
        logger.info(f"   - PDFs processed: {len(pdf_files)}")
        logger.info(f"   - Image texts extracted: {len(all_extracted_data)}")
        logger.info(f"   - Total characters: {total_text_chars}")
        logger.info(f"   - Output file: {output_path}")
        
        return all_extracted_data

def main():
    parser = argparse.ArgumentParser(description="Selective Image Text Extraction")
    parser.add_argument("--input", "-i", default=DEFAULT_LOCAL_DIR,
                       help=f"Input PDF file or directory containing PDFs (default: {DEFAULT_LOCAL_DIR})")
    parser.add_argument("--output", "-o", default="extracted_image_texts.json",
                       help="Output JSON file (default: extracted_image_texts.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = SelectiveImageProcessor()
    processor.process_pdfs(args.input, args.output)

if __name__ == "__main__":
    main()