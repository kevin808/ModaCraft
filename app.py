import streamlit as st
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
import httpx
import json
import logging
import time
import asyncio
from config import CONFIG
from openai import OpenAI

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI client setup
llm_client = OpenAI(
    api_key=CONFIG["API_KEY"],
    base_url= CONFIG["BASE_URL"]
)

# SearxNG client setup
SEARXNG_URL = CONFIG["SEARXNG_URL"] # Update with your SearxNG instance URL

async def search_content(query: str) -> Dict:
    logging.debug(f"Initiating search for query: {query}")
    try:
        async with httpx.AsyncClient() as client:
            logging.debug(f"Sending request to SearxNG: {SEARXNG_URL}/search")
            response = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": ["general","news"],
                }
            )
            response.raise_for_status()  # Raise exception for non-200 status codes
            try:
                data = response.json()
                logging.debug(f"!!!!!!!Search response: {data} !!!!!!!")
                logging.debug(f"Search returned {len(data.get('results', []))} results")
                return data
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response: {response.text}")
                return {"results": []}
    except Exception as e:
        logging.error(f"Search failed: {str(e)}")
        return {"results": []}

class ContentItem(BaseModel):
    type: str  # text, image, video, audio
    url: Optional[HttpUrl]
    content: Optional[str]
    title: Optional[str]
    source: Optional[str]

# 1. 定义MCP Input协议数据结构
class MCPInputData(BaseModel):
    text: Optional[str] = None
    image_urls: Optional[List[HttpUrl]] = []
    retrieved_content: Optional[List[ContentItem]] = []

class MCPRequest(BaseModel):
    command: str                # 用户自然语言指令
    modalities: MCPInputData    # 多模态数据集合

# 2. 定义MCP Output协议数据结构
class MCPOutputData(BaseModel):
    result_text: str    # 生成结果
    used_images: List[HttpUrl]
    references: List[ContentItem] = []

class MCPResponse(BaseModel):
    status: str
    response: MCPOutputData

# LLM API call with retry logic
async def call_llm(messages, model="gpt-4o", temperature=0.7, max_retries=3):
    retry_count = 0
    backoff_time = 1
    
    while True:
        try:
            logging.debug(f"Calling API with model={model}, temperature={temperature}")
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            logging.debug("API call successful")
            return response.choices[0].message.content
                
        except Exception as e:
            logging.error(f"Error calling API: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Max retries ({max_retries}) reached, giving up")
                raise
            logging.warning(f"API call failed: {e}. Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff

# 3. 示例“多模态大模型接口”逻辑（mock模拟）
async def call_multimodal_llm(command: str, modalities: MCPInputData):
    logging.debug(f"Processing multimodal request with command: {command}")
    
    # 1. Search for relevant content
    logging.debug("Starting content search")
    search_results = await search_content(command)
    
    # 2. Process and organize retrieved content
    logging.debug("Processing search results")
    retrieved_items = []
    for result in search_results.get("results", []):
        content_type = "text"
        if result.get("img_src"):
            content_type = "image"
        elif result.get("video_url"):
            content_type = "video"
            
        retrieved_items.append(ContentItem(
            type=content_type,
            url=result.get("url"),
            title=result.get("title"),
            content=result.get("content"),
            source=result.get("source")
        ))
    
    logging.debug(f"Processed {len(retrieved_items)} content items")
    modalities.retrieved_content = retrieved_items

    # 3. Prepare content for LLM
    context = "\n".join([
        f"[{idx+1}] {item.title}: {item.content}" 
        for idx, item in enumerate(retrieved_items) 
        if item.content
    ])
    
    logging.debug("Preparing LLM prompt")
    prompt = f"""基于以下搜索到的信息，请总结分析关于"{command}"的情况：

{context}

请给出一个结构化的分析报告。在引用信息时，请使用方括号中的数字来标注引用来源，例如[1]、[2]等。"""
    
    logging.debug(f"LLM prompt: {prompt}")

    # 4. Call LLM API
    try:
        logging.debug("Calling LLM for analysis")
        result_text = await call_llm(
            messages=[
                {"role": "system", "content": "你是一个专业的多模态内容分析助手，善于整合各类信息并给出深入分析。"},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o"
        )
    except Exception as e:
        logging.error(f"LLM analysis failed: {str(e)}")
        result_text = f"API调用出错: {str(e)}"
    
    logging.debug("Preparing final response")
    return MCPOutputData(
        result_text=result_text,
        used_images=[item.url for item in retrieved_items if item.type == "image"][:2],
        references=retrieved_items[:5]
    )

def main():
    st.title("ModaCraft - 多模态内容助手")
    st.write("输入您的问题，我们将为您搜索和分析相关内容")

    # User input
    query = st.text_input("请输入您的问题：")
    
    if st.button("分析"):
        if query:
            with st.spinner('正在处理您的请求...'):
                # Create input data structure
                input_data = MCPInputData(text=query)
                
                # Process the request
                result = asyncio.run(call_multimodal_llm(query, input_data))
                
                # Display results
                st.subheader("分析结果")
                st.write(result.result_text)
                
                if result.used_images:
                    st.subheader("相关图片")
                    for img_url in result.used_images:
                        st.image(str(img_url))
                
                if result.references:
                    st.subheader("参考来源")
                    for idx, ref in enumerate(result.references, 1):
                        with st.expander(f"[{idx}] {ref.title or '参考内容'}"):
                            if ref.content:
                                st.write(ref.content)
                            if ref.url:
                                st.write(f"链接: {ref.url}")

if __name__ == "__main__":
    main()