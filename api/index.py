import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from tavily import TavilyClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    product_name: str

@app.post("/api/search")
async def search_product(req: SearchRequest):
    if not req.product_name or not req.product_name.strip():
        raise HTTPException(status_code=400, detail="กรุณากรอกชื่อสินค้า")

    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not groq_key:
        raise HTTPException(status_code=500, detail="ยังไม่ได้ใส่ GROQ_API_KEY บน Vercel")
    if not tavily_key:
        raise HTTPException(status_code=500, detail="ยังไม่ได้ใส่ TAVILY_API_KEY บน Vercel")

    # 1. ดึงข้อมูล Tavily (ทำงานผ่านแล้วจาก Log)
    raw_context = ""
    try:
        tavily = TavilyClient(api_key=tavily_key)
        search_res = tavily.search(
            query=f"{req.product_name} ราคา สเปก รีวิว ข้อเสีย ปัญหา Pantip",
            search_depth="basic",
            max_results=4
        )
        results = search_res.get("results", [])
        if results:
            raw_context = "\n".join([f"- {r.get('title', '')}: {r.get('content', '')}" for r in results])
    except Exception as e:
        print(f"Tavily Error: {e}")
        raw_context = "ไม่สามารถดึงข้อมูลสดได้ ให้ใช้วิเคราะห์ตามฐานข้อมูลของคุณ"

    # 2. ส่งให้ Groq วิเคราะห์และแปลงผล
    try:
        client = Groq(api_key=groq_key)
        
        prompt = f"""
        คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์สินค้า จงนำข้อมูลของ "{req.product_name}" ต่อไปนี้มาสรุปในรูปแบบ JSON เท่านั้น:

        [ข้อมูลสดจากเว็บ]:
        {raw_context}

        โปรดตอบกลับเป็น JSON Structure ตามนี้เท่านั้น:
        {{
            "price_summary": "สรุปช่วงราคาล่าสุด (เช่น 35,900 - 42,000 บาท)",
            "specs_and_details": ["จุดเด่นที่ 1", "จุดเด่นที่ 2", "จุดเด่นที่ 3"],
            "community_reviews": {{
                "positive": ["ข้อดี 1", "ข้อดี 2"],
                "negative": ["ข้อเสีย 1", "ข้อเสีย 2"]
            }},
            "drama_and_trends": "สรุปกระแส ดราม่า หรือข้อควรระวัง"
        }}
        """

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        
        # คลีนข้อความขยะ/สัญลักษณ์ครอบป้องการแปลง JSON พัง
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        return json.loads(content)

    except Exception as e:
        print(f"Groq/JSON Error: {e}")
        raise HTTPException(status_code=500, detail=f"Backend Processing Error: {str(e)}")
        
