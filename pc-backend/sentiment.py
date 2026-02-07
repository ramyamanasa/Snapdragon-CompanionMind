from datetime import datetime
from typing import Dict, List

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Expanded keyword categories
        self.emotion_keywords = {
            "loneliness": [
                "lonely", "alone", "isolated", "forgotten", "nobody",
                "no one", "by myself", "on my own", "miss", "missing",
                "abandoned", "left behind", "empty", "solitary"
            ],
            "sadness": [
                "sad", "depressed", "down", "blue", "unhappy",
                "miserable", "hopeless", "despair", "crying", "tears",
                "heartbroken", "grief", "sorrow"
            ],
            "anxiety": [
                "worried", "anxious", "scared", "afraid", "fear",
                "nervous", "stress", "panic", "overwhelmed", "restless"
            ],
            "social_disconnection": [
                "nobody calls", "nobody visits", "don't talk to anyone",
                "haven't heard from", "they forgot", "too busy for me",
                "don't care", "ignored", "excluded"
            ]
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Comprehensive sentiment analysis
        Returns detailed emotion breakdown with confidence scores
        """
        text_lower = text.lower()
        
        # Analyze each emotion category
        emotion_scores = {}
        detected_keywords = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            
            if matches:
                # Calculate confidence based on number of matches
                confidence = min(len(matches) * 25, 100)  # Max 100%
                emotion_scores[emotion] = confidence
                detected_keywords[emotion] = matches
        
        # Determine primary emotion
        primary_emotion = None
        max_confidence = 0
        
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            max_confidence = emotion_scores[primary_emotion]
        
        # Calculate overall negativity score
        total_matches = sum(len(kw) for kw in detected_keywords.values())
        negativity_score = min(total_matches * 15, 100)
        
        # Determine severity
        if negativity_score >= 60:
            severity = "severe"
        elif negativity_score >= 30:
            severity = "moderate"
        elif negativity_score > 0:
            severity = "mild"
        else:
            severity = "none"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "primary_emotion": primary_emotion,
            "confidence": max_confidence,
            "severity": severity,
            "negativity_score": negativity_score,
            "emotion_breakdown": emotion_scores,
            "detected_keywords": detected_keywords,
            "needs_attention": negativity_score >= 30
        }
    
    def compute_trend_and_risk(self, sentiment_history):
        if len(sentiment_history) < 4:
            return {
                "trend": "insufficient_data",
                "risk_level": "low"
            }

        # Recent vs older negativity
        recent = sentiment_history[-3:]
        older = sentiment_history[-6:-3]

        recent_avg = sum(s.get("negativity_score", 0) for s in recent) / len(recent)
        older_avg = sum(s.get("negativity_score", 0) for s in older) / len(older)

        # Determine trend
        if recent_avg > older_avg + 10:
            trend = "worsening"
        elif recent_avg < older_avg - 10:
            trend = "improving"
        else:
            trend = "stable"

        # Determine risk level
        if recent_avg >= 60:
            risk = "high"
        elif recent_avg >= 30:
            risk = "medium"
        else:
            risk = "low"

        return {
            "trend": trend,
            "risk_level": risk,
            "recent_avg_negativity": round(recent_avg, 2)
    }

    
    def analyze_pattern(self, sentiment_history: List[Dict]) -> Dict:
        """
        Detect concerning patterns over time
        """
        if len(sentiment_history) < 3:
            return {
                "pattern_detected": False,
                "reason": "Insufficient data (need 3+ conversations)"
            }
        
        # Get recent sentiments (last 10)
        recent = sentiment_history[-10:]
        
        # Count high-negativity instances
        high_negativity_count = sum(
            1 for s in recent 
            if s.get("negativity_score", 0) >= 30
        )
        
        # Count loneliness mentions
        loneliness_count = sum(
            1 for s in recent 
            if s.get("primary_emotion") == "loneliness"
        )
        
        # Detect escalation (scores getting worse)
        if len(recent) >= 5:
            recent_avg = sum(s.get("negativity_score", 0) for s in recent[-3:]) / 3
            older_avg = sum(s.get("negativity_score", 0) for s in recent[:3]) / 3
            escalating = recent_avg > older_avg + 20
        else:
            escalating = False
        
        # Trigger alerts
        alert_triggered = False
        alert_reason = []
        
        if high_negativity_count >= 3:
            alert_triggered = True
            alert_reason.append(f"{high_negativity_count} negative conversations detected")
        
        if loneliness_count >= 2:
            alert_triggered = True
            alert_reason.append(f"Loneliness mentioned {loneliness_count} times")
        
        if escalating:
            alert_triggered = True
            alert_reason.append("Emotional state appears to be worsening")
        
        return {
            "pattern_detected": alert_triggered,
            "severity": "high" if high_negativity_count >= 5 else "moderate",
            "reasons": alert_reason,
            "high_negativity_count": high_negativity_count,
            "loneliness_mentions": loneliness_count,
            "escalating": escalating,
            "recommendation": "Consider reaching out for a personal check-in" if alert_triggered else "Continue monitoring"
        }
    
    
    
    