import aiohttp
import asyncio

def run_async(func):
    def wrapper(*args):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(func(*args))
        return result
    return wrapper

@run_async
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return response, await response.json()

@run_async 
async def post_data(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            return response, await response.json()

@run_async      
async def post_data_predict(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            try:
                # Пробуем распарсить ответ как JSON
                return response, await response.json()
            except aiohttp.ContentTypeError:
                # Если не удается распарсить как JSON, возвращаем текст
                return response, await response.text()

@run_async
async def delete_id_model(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.delete(url, data=data, headers={'Content-Type': 'application/json'}) as response:
            return response, await response.json()

@run_async
async def delete_models(url):
    async with aiohttp.ClientSession() as session:
        async with session.delete(url) as response:
            return response, await response.json()

