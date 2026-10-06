import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# เรียกใช้ Gemini API Client (จะดึง GEMINI_API_KEY จาก Vercel Environment Variables)
client = genai.Client()

# Schema สำหรับรับข้อมูลจาก Frontend
class SearchRequest(BaseModel):
    product_name: str

@app.post("/api/search")
async def search_product(req: SearchRequest):
    if not req.product_name.strip():
        raise HTTPException(status_code=400, detail="Product name is required")

    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์สินค้าและกระแสโซเชียล
    กรุณาค้นหาและสรุปข้อมูลล่าสุดของสินค้าชื่อ: "{req.product_name}"
    
    โดยส่งกลับข้อมูลมาในรูปแบบ JSON ตามโครงสร้างนี้เท่านั้น (ห้ามใส่ Markdown code block หรือข้อความอื่นปน):
    {{
        "price_summary": "สรุปช่วงราคาล่าสุด (เช่น 35,900 - 42,000 บาท)",
        "specs_and_details": [
            "สรุปสเปกหรือจุดเด่นข้อที่ 1",
            "สรุปสเปกหรือจุดเด่นข้อที่ 2",
            "สรุปสเปกหรือจุดเด่นข้อที่ 3"
        ],
        "community_reviews": {{
            "positive": ["ข้อดีหรือจุดที่คนชม 1", "ข้อดีหรือจุดที่คนชม 2"],
            "negative": ["ข้อเสียหรือจุดสังเกต 1", "ข้อเสียหรือจุดสังเกต 2"]
        }},
        "drama_and_trends": "สรุปกระแสดราม่า ปัญหาที่พบบ่อย หรือสิ่งที่ควรระวังจากผู้ใช้จริงในโซเชียล"
    }}
    """

    try:
        # เรียกใช้ Gemini 2.5 Flash พร้อมเปิดการใช้ Google Search (Grounding)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                response_mime_type="application/json"
            )
        )

        # แปลงข้อความที่ Gemini ตอบกลับมาเป็น JSON
        result_data = json.loads(response.text)
        return result_data

    except Exception as e:
        print(f"Error generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch product insights")
