import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from duckduckgo_search import DDGS

app = FastAPI()

# เปิดอนุญาต CORS ให้ GitHub Pages ยิงข้ามโดเมนเข้ามาดึงข้อมูลได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_product_data(product_name: str) -> dict:
    price_info = []
    review_info = []
    
    try:
        with DDGS() as ddgs:
            # ค้นหาราคาและสเปก
            price_results = list(ddgs.text(f"{product_name} ราคา สเปก", region="th-th", max_results=3))
            for r in price_results:
                price_info.append(f"- {r.get('title', '')}: {r.get('body', '')}")
                
            # ค้นหารีวิว
            review_results = list(ddgs.text(f"{product_name} รีวิว ข้อเสีย ปัญหา", region="th-th", max_results=3))
            for r in review_results:
                review_info.append(f"- {r.get('title', '')}: {r.get('body', '')}")
    except Exception as e:
        print(f"[Search Warning]: {e}")

    return {
        "prices": "\n".join(price_info) if price_info else "ไม่สามารถดึงข้อมูลสดจากเว็บได้ ให้ใช้วิเคราะห์จากฐานข้อมูลของคุณ",
        "reviews": "\n".join(review_info) if review_info else "ไม่สามารถดึงรีวิวสดจากเว็บได้ ให้ใช้วิเคราะห์จากฐานข้อมูลของคุณ"
    }

class SearchRequest(BaseModel):
    product_name: str

@app.post("/api/search")
async def search_product(req: SearchRequest):
    if not req.product_name.strip():
        raise HTTPException(status_code=400, detail="Product name is required")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set on Vercel")

    client = Groq(api_key=api_key)
    raw_data = fetch_product_data(req.product_name)
    
    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์สินค้า จงนำข้อมูลของ "{req.product_name}" ต่อไปนี้มาวิเคราะห์และสรุปให้อยู่ในรูปแบบ JSON เท่านั้น:
    
    [ข้อมูลราคาและสเปกดิบ]:
    {raw_data['prices']}
    
    [ข้อมูลรีวิวและกระแสโซเชียลดิบ]:
    {raw_data['reviews']}
    
    ตอบกลับเฉพาะ JSON โครงสร้างนี้เท่านั้น (ห้ามมีคำเกริ่นหรือข้อความ markdown เช่น ```json ปนเด็ดขาด):
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
        # เปลี่ยนเป็นโมเดล Llama 3.3 70B ที่เสถียรที่สุดของ Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print(f"Error generation: {e}")
        # ส่งข้อความ Error ที่แท้จริงออกมาเพื่อให้เช็กง่ายขึ้น
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")
    
