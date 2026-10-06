import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from tavily import TavilyClient

app = FastAPI()

# เปิดอนุญาต CORS ให้ GitHub Pages ยิงข้ามโดเมนเข้ามาดึงข้อมูลได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_product_data_tavily(product_name: str, tavily_key: str) -> dict:
    price_info = []
    review_info = []
    
    try:
        tavily = TavilyClient(api_key=tavily_key)
        
        # 1. ค้นหาราคาและสเปกเรียลไทม์
        price_res = tavily.search(
            query=f"{product_name} ราคา สเปก ซื้อที่ไหน",
            search_depth="basic",
            max_results=3
        )
        for r in price_res.get("results", []):
            price_info.append(f"- {r.get('title')}: {r.get('content')}")

        # 2. ค้นหารีวิว ข้อเสีย และดราม่าเรียลไทม์
        review_res = tavily.search(
            query=f"{product_name} รีวิว ข้อเสีย ปัญหา ดราม่า Pantip Facebook",
            search_depth="basic",
            max_results=3
        )
        for r in review_res.get("results", []):
            review_info.append(f"- {r.get('title')}: {r.get('content')}")

    except Exception as e:
        print(f"[Tavily Search Warning]: {e}")

    return {
        "prices": "\n".join(price_info) if price_info else "ไม่สามารถดึงข้อมูลสดได้ ให้วิเคราะห์ตามฐานข้อมูลของคุณ",
        "reviews": "\n".join(review_info) if review_info else "ไม่สามารถดึงรีวิวสดได้ ให้วิเคราะห์ตามฐานข้อมูลของคุณ"
    }

class SearchRequest(BaseModel):
    product_name: str

@app.post("/api/search")
async def search_product(req: SearchRequest):
    if not req.product_name.strip():
        raise HTTPException(status_code=400, detail="Product name is required")

    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set on Vercel")
    if not tavily_key:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY is not set on Vercel")

    client = Groq(api_key=groq_key)
    raw_data = fetch_product_data_tavily(req.product_name, tavily_key)
    
    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์สินค้า จงนำข้อมูลเรียลไทม์สดๆ ของ "{req.product_name}" ต่อไปนี้มาวิเคราะห์และสรุปให้อยู่ในรูปแบบ JSON เท่านั้น:
    
    [ข้อมูลราคาและสเปกสดจากเว็บ]:
    {raw_data['prices']}
    
    [ข้อมูลรีวิวและกระแสโซเชียลสดจากเว็บ]:
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print(f"Error generation: {e}")
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")
    
