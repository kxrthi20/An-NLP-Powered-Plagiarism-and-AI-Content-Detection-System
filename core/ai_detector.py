# Minimal AI Detector Heuristics
# We use sentence length variation (Burstiness) and repeated N-grams (predictability) to estimate AI vs Human.
# AI-generated text is often overly consistent and lacks major sentence length variations.

import textwrap
import re

class AIOstDetector:
    def detect(self, text: str) -> dict:
        if not text.strip():
            return {"ai_probability": 0, "human_probability": 100, "label": "Too short"}

        # Count sentences roughly by punctuation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        # Calculate length based heuristic for any amount of text
        words = text.split()
        if len(words) < 5:
            # For extremely short texts, just use length hash for deterministic pseudo-result
            ai_score = 40 + (len(text) % 30)
            return {
                "ai_probability": ai_score,
                "human_probability": 100 - ai_score,
                "label": "Mixed / Unclear"
            }

        # Calculate burstiness (variance of sentence/phrase lengths)
        lengths = [len(s.split()) for s in sentences] if len(sentences) > 0 else [len(words)]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

        # AI tends to have low variance (consistent sentence length). Humans have high variance.
        # We'll normalize this into a heuristic score.
        
        # Assume a baseline human variance around 40-70. AI variance often below 20.
        ai_score = 100 - min(100, variance * 2) 

        # Add a bit of randomness to the heuristic based on vocabulary (just for visual representation logic)
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        lexical_density = unique_words / total_words if total_words > 0 else 0
        
        # AI models sometimes use highly standard vocabulary resulting in middle-ground density.
        if 0.4 < lexical_density < 0.6:
            ai_score += 15

        ai_score = max(0, min(100, int(ai_score)))
        human_score = 100 - ai_score

        if ai_score >= 65:
            label = "Likely AI Generated"
        elif ai_score <= 35:
            label = "Likely Human Written"
        else:
            label = "Mixed / Unclear"

        return {
            "ai_probability": ai_score,
            "human_probability": human_score,
            "label": label
        }

if __name__ == "__main__":
    import sys
    detector = AIOstDetector()
    sample_text = "This is a sample text to test the AI detector. AI-generated text is often overly consistent and lacks major sentence length variations."
    
    print("--- Testing AIOstDetector ---")
    print(f"Sample Text: {sample_text}")
    print("\nResult:")
    result = detector.detect(sample_text)
    for key, value in result.items():
        print(f"  {key}: {value}")
