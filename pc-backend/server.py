from sentiment import EnhancedSentimentAnalyzer
from typing import List, Dict
import json
import subprocess
from datetime import datetime
import os
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="CompanionMind AI Engine")

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

phone_path = os.path.join(PARENT_DIR, "phone")
dashboard_path = os.path.join(PARENT_DIR, "dashboard")

if os.path.exists(phone_path):
    from fastapi.responses import FileResponse
    
    @app.get("/phone/{file_path:path}")
    async def serve_phone(file_path: str):
        full_path = os.path.join(phone_path, file_path)
        return FileResponse(
            full_path, 
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    
    print(f"✓ Serving phone app from: {phone_path} (no-cache)")

if os.path.exists(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")
    print(f"✓ Serving dashboard from: {dashboard_path}")

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Storage
# ---------------------------
conversations = []

user_profile = {
    "name": "User",
    "loneliness_mentions": 0,
    "sentiment_scores": []
}

# Sensor data storage
sensor_data = {
    "motion": {
        "current_activity": "unknown",
        "steps_today": 0,
        "last_movement": None,
        "fall_alerts": [],
        "activity_history": []
    },
    "location": {
        "is_home": True,
        "left_home_today": False,
        "last_update": None,
        "history": []
    },
    "light": {
        "current_level": None,
        "is_dark": False,
        "dark_duration_minutes": 0,
        "last_update": None,
        "history": []
    }
}

# ===========================
# AI ENGINE
# ===========================
class AIEngine:
    def __init__(self):
        self.model = "phi3:mini"
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()

    def get_trend_and_risk(self):
        if len(user_profile["sentiment_scores"]) < 4:
            return {
                "trend": "insufficient_data",
                "risk_level": "low",
                "recent_avg_negativity": 0
            }

        return self.sentiment_analyzer.compute_trend_and_risk(
            user_profile["sentiment_scores"]
        )

    def get_combined_risk_assessment(self):
        """Combines emotional sentiment with physical sensor data"""
        emotional_risk = self.get_trend_and_risk()
        
        risk_factors = []
        risk_score = 0
        
        # Check motion/activity
        if sensor_data["motion"]["steps_today"] < 500:
            risk_factors.append("Very low physical activity (< 500 steps)")
            risk_score += 2
        
        # Check location
        if not sensor_data["location"]["left_home_today"]:
            risk_factors.append("Has not left home today")
            risk_score += 1
        
        # Check fall alerts
        if len(sensor_data["motion"]["fall_alerts"]) > 0:
            risk_factors.append(f"{len(sensor_data['motion']['fall_alerts'])} fall alert(s) detected")
            risk_score += 3
        
        # Emotional risk score
        emotional_score = 0
        if emotional_risk["risk_level"] == "high":
            emotional_score = 3
        elif emotional_risk["risk_level"] == "medium":
            emotional_score = 2
        
        total_risk_score = risk_score + emotional_score
        
        if total_risk_score >= 6:
            overall_risk = "critical"
        elif total_risk_score >= 4:
            overall_risk = "high"
        elif total_risk_score >= 2:
            overall_risk = "moderate"
        else:
            overall_risk = "low"
        
        return {
            "overall_risk_level": overall_risk,
            "risk_score": total_risk_score,
            "emotional_component": emotional_risk,
            "physical_risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(overall_risk, risk_factors)
        }
    
    def _get_risk_recommendation(self, risk_level, factors):
        if risk_level == "critical":
            return "IMMEDIATE ACTION REQUIRED: Contact user immediately."
        elif risk_level == "high":
            return "HIGH PRIORITY: Reach out to user soon."
        elif risk_level == "moderate":
            return "MONITOR CLOSELY: Continue observation."
        else:
            return "ROUTINE MONITORING: User appears stable."

    def generate_response(self, user_message: str, history: list) -> str:
        """Generate AI response with aggressive cleaning"""
        
        context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in history[-5:]
        ])

        # SIMPLIFIED PROMPT - less likely to leak
        prompt = f"""You are a caring companion for an elderly person. Be warm and brief.

Conversation:
{context}

User: {user_message}
Assistant:"""

        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            response = result.stdout.strip()

            if not response:
                return "I'm here with you. Tell me more."

            # AGGRESSIVE CLEANING
            # Remove everything after certain markers
            stop_markers = [
                '\n---', '\n###', 'Instruction', 'Note:', 'Consider',
                'Let us', 'System:', 'scenario', 'situation where',
                'Example:', 'User:', 'Assistant:', 'Human:', 'AI:'
            ]
            
            for marker in stop_markers:
                if marker in response:
                    response = response.split(marker)[0].strip()
            
            # Split into lines and only keep actual response
            lines = response.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Stop at instruction-like text
                if any(word in line.lower() for word in ['instruction', 'note:', 'consider', 'let us', 'scenario']):
                    break
                clean_lines.append(line)
            
            response = ' '.join(clean_lines).strip()
            
            # Limit to 2 sentences
            sentences = response.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 2:
                response = '. '.join(sentences[:2]) + '.'
            elif sentences:
                response = '. '.join(sentences)
                if not response.endswith(('.', '!', '?')):
                    response += '.'
            
            return response if response else "I'm here listening."

        except Exception as e:
            print(f"❌ AI Error: {e}")
            return "I'm here with you."

    def analyze_sentiment(self, text: str) -> dict:
        return self.sentiment_analyzer.analyze(text)

    def analyze_patterns(self, scores: List[Dict]) -> Dict:
        return self.sentiment_analyzer.analyze_pattern(scores)


ai_engine = AIEngine()

# ===========================
# SENSOR DATA HANDLERS
# ===========================

def handle_motion_update(data):
    """Process motion sensor data from phone"""
    sensor_data["motion"]["current_activity"] = "active" if data.get("isActive") else "sedentary"
    sensor_data["motion"]["steps_today"] = data.get("steps", 0)
    sensor_data["motion"]["last_movement"] = data.get("lastMovement")
    
    activity_snapshot = {
        "timestamp": data.get("timestamp"),
        "is_active": data.get("isActive"),
        "movement_count": data.get("movementCount"),
        "steps": data.get("steps", 0)
    }
    
    sensor_data["motion"]["activity_history"].append(activity_snapshot)
    
    if len(sensor_data["motion"]["activity_history"]) > 100:
        sensor_data["motion"]["activity_history"].pop(0)
    
    print(f"🏃 Motion Update: {sensor_data['motion']['current_activity']} | "
          f"Steps: {data.get('steps', 0)} | "
          f"Movements: {data.get('movementCount')}")

def handle_fall_alert(data):
    """Handle fall detection alert from phone"""
    fall_event = {
        "timestamp": data.get("timestamp"),
        "magnitude": data.get("magnitude"),
        "time": datetime.fromtimestamp(data.get("timestamp") / 1000).isoformat()
    }
    
    sensor_data["motion"]["fall_alerts"].append(fall_event)
    
    if len(sensor_data["motion"]["fall_alerts"]) > 50:
        sensor_data["motion"]["fall_alerts"].pop(0)
    
    print(f"\n{'='*60}")
    print(f"🚨 FALL ALERT DETECTED!")
    print(f"   Magnitude: {data.get('magnitude')} m/s²")
    print(f"   Time: {fall_event['time']}")
    print(f"   CAREGIVER NOTIFICATION TRIGGERED")
    print(f"{'='*60}\n")

def handle_location_update(data):
    """Process GPS location data from phone"""
    sensor_data["location"]["is_home"] = data.get("isHome", True)
    sensor_data["location"]["left_home_today"] = data.get("leftHomeToday", False)
    sensor_data["location"]["last_update"] = data.get("timestamp")
    
    location_snapshot = {
        "timestamp": data.get("timestamp"),
        "is_home": data.get("isHome"),
        "distance_from_home": data.get("distance", 0)
    }
    
    sensor_data["location"]["history"].append(location_snapshot)
    
    if len(sensor_data["location"]["history"]) > 100:
        sensor_data["location"]["history"].pop(0)
    
    status = "at home" if data.get("isHome") else "away from home"
    print(f"📍 Location Update: {status} | "
          f"Distance: {data.get('distance', 0)}m | "
          f"Left home today: {data.get('leftHomeToday')}")

def handle_light_update(data):
    """Process ambient light sensor data from phone"""
    sensor_data["light"]["current_level"] = data.get("currentLevel")
    sensor_data["light"]["is_dark"] = data.get("isDark", False)
    sensor_data["light"]["dark_duration_minutes"] = data.get("darkDuration", 0)
    sensor_data["light"]["last_update"] = data.get("timestamp")
    
    light_snapshot = {
        "timestamp": data.get("timestamp"),
        "level": data.get("currentLevel"),
        "is_dark": data.get("isDark")
    }
    
    if "history" not in sensor_data["light"]:
        sensor_data["light"]["history"] = []
    
    sensor_data["light"]["history"].append(light_snapshot)
    
    if len(sensor_data["light"]["history"]) > 100:
        sensor_data["light"]["history"].pop(0)
    
    dark_status = "DARK" if data.get("isDark") else "bright"
    print(f"💡 Light Update: {data.get('currentLevel')} lux ({dark_status}) | "
          f"Dark duration: {data.get('darkDuration')} min")

# ===========================
# ROOT
# ===========================
@app.get("/")
async def home():
    return {
        "service": "CompanionMind AI Engine",
        "status": "running",
        "model": "Phi-3 Mini (Snapdragon Optimized)",
        "conversations": len(conversations) // 2,
        "sensors_active": {
            "motion": sensor_data["motion"]["last_movement"] is not None,
            "location": sensor_data["location"]["last_update"] is not None,
            "light": sensor_data["light"]["last_update"] is not None
        }
    }

# ===========================
# STATS (Dashboard)
# ===========================
@app.get("/stats")
async def get_stats():

    recent = user_profile["sentiment_scores"][-10:]
    avg_negativity = 0

    if recent:
        avg_negativity = sum(s.get("negativity_score", 0) for s in recent) / len(recent)

    pattern_analysis = {}
    if len(user_profile["sentiment_scores"]) >= 3:
        pattern_analysis = ai_engine.analyze_patterns(user_profile["sentiment_scores"])

    trend_info = ai_engine.get_trend_and_risk()
    combined_risk = ai_engine.get_combined_risk_assessment()
    
    # Calculate activity percentage
    recent_activity = sensor_data["motion"]["activity_history"][-20:]
    active_count = sum(1 for a in recent_activity if a.get("is_active"))
    activity_percentage = (active_count / len(recent_activity) * 100) if recent_activity else 0

    return {
        "total_conversations": len(conversations) // 2,
        "loneliness_mentions": user_profile["loneliness_mentions"],
        "average_negativity": round(avg_negativity, 2),
        "recent_sentiments": recent,
        "pattern_alert": pattern_analysis,
        "trend": trend_info,
        "last_conversation": conversations[-1]["timestamp"] if conversations else None,
        
        # Sensor data
        "sensors": {
            "motion": {
                "activity_status": sensor_data["motion"]["current_activity"],
                "steps_today": sensor_data["motion"]["steps_today"],
                "activity_percentage": round(activity_percentage, 1),
                "last_movement": sensor_data["motion"]["last_movement"],
                "fall_alerts": sensor_data["motion"]["fall_alerts"][-10:],
                "recent_activity": recent_activity
            },
            "location": {
                "is_home": sensor_data["location"]["is_home"],
                "left_home_today": sensor_data["location"]["left_home_today"],
                "last_update": sensor_data["location"]["last_update"],
                "recent_history": sensor_data["location"]["history"][-10:]
            },
            "light": {
                "current_level": sensor_data["light"]["current_level"],
                "is_dark": sensor_data["light"]["is_dark"],
                "dark_duration_minutes": sensor_data["light"]["dark_duration_minutes"],
                "last_update": sensor_data["light"]["last_update"]
            }
        },
        
        # Combined risk assessment
        "combined_risk": combined_risk
    }

# ===========================
# WEBSOCKET (Phone)
# ===========================
@app.websocket("/ws/client")
async def websocket_client(websocket: WebSocket):
    await websocket.accept()
    print("\n" + "="*60)
    print("✓ Phone connected to WebSocket!")
    print("="*60)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")
            
            # ===========================
            # SENSOR MESSAGE HANDLERS
            # ===========================
            if msg_type == "motion_update":
                handle_motion_update(data)
                continue
                
            elif msg_type == "fall_alert":
                handle_fall_alert(data)
                await websocket.send_json({
                    "type": "fall_alert_received",
                    "message": "Fall alert received and logged."
                })
                continue
                
            elif msg_type == "location_update":
                handle_location_update(data)
                continue
                
            elif msg_type == "light_update":
                handle_light_update(data)
                continue
            
            # ===========================
            # REGULAR CONVERSATION
            # ===========================
            user_message = data.get("content", "").strip()

            if not user_message:
                continue

            print(f"\n📱 User: {user_message}")

            conversations.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })

            # Sentiment analysis
            sentiment = ai_engine.analyze_sentiment(user_message)
            user_profile["sentiment_scores"].append(sentiment)

            if sentiment.get("primary_emotion"):
                print(f"⚠️  Emotion: {sentiment['primary_emotion']} ({sentiment['confidence']}%)")
                print(f"   Severity: {sentiment['severity']} | Negativity: {sentiment['negativity_score']}/100")

            if sentiment.get("primary_emotion") == "loneliness":
                user_profile["loneliness_mentions"] += 1
                print(f"🚨 Loneliness mention count: {user_profile['loneliness_mentions']}")

            pattern_analysis = ai_engine.analyze_patterns(user_profile["sentiment_scores"])

            if pattern_analysis.get("pattern_detected"):
                print(f"\n🚨 PATTERN ALERT!")
                print(f"   Severity: {pattern_analysis.get('severity')}")
                print(f"   Reasons: {', '.join(pattern_analysis.get('reasons', []))}")

            # AI response
            ai_response = ai_engine.generate_response(user_message, conversations)

            conversations.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })

            print(f"🤖 AI: {ai_response}")
            print("="*60 + "\n")

            trend_info = ai_engine.get_trend_and_risk()
            combined_risk = ai_engine.get_combined_risk_assessment()

            await websocket.send_json({
                "type": "response",
                "content": ai_response,
                "metadata": {
                    "sentiment": sentiment,
                    "pattern_alert": pattern_analysis,
                    "trend": trend_info,
                    "combined_risk": combined_risk,
                    "timestamp": datetime.now().isoformat()
                }
            })

    except WebSocketDisconnect:
        print("\n📱 Phone disconnected from WebSocket")
    except Exception as e:
        print(f"\n❌ WebSocket Error: {e}")
        traceback.print_exc()

# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 CompanionMind AI Engine - Samsung Internet Optimized")
    print("="*60)
    print(f"🤖 AI Model: Phi-3 Mini (Snapdragon NPU)")
    print(f"📊 Sentiment Analysis: Enhanced Multi-Emotion")
    print(f"📱 Sensor Support: Motion, Location")
    print(f"🔗 WebSocket: Real-time Communication")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")