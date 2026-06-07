import os
from core.plagiarism import PlagiarismDetector
from core.ai_detector import AIOstDetector

def main():
    print("--- Plagiarism & AI Detector Test Runner ---")
    
    docA = "This is a simple text document meant to establish a baseline for plagiarism detection."
    docB = "This is a simple text document meant to establish a baseline for plagiarism detection. It has slightly different words at the end."
    
    print("\n[Document A]:", docA)
    print("\n[Document B]:", docB)
    
    print("\n--- Running Plagiarism Detection ---")
    p_detector = PlagiarismDetector()
    results = p_detector.detect_plagiarism(docA, docB)
    stats = p_detector.get_word_stats(docA, docB)
    print(f"Similarity: {results['score']}%")
    print(f"Level: {results['level']}")
    print(f"Top Common Words: {stats['common_words']}")
    
    print("\n--- Running AI Content Detection ---")
    ai_detector = AIOstDetector()
    ai_results = ai_detector.detect(docA)
    print(f"AI Probability: {ai_results['ai_probability']}%")
    print(f"Human Probability: {ai_results['human_probability']}%")
    print(f"Label: {ai_results['label']}")

if __name__ == "__main__":
    main()
