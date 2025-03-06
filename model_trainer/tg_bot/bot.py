import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from config import BOT
from handlers import router
from middlewares import LoggingMiddleware

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT)
storage = MemoryStorage()

dp = Dispatcher(storage=storage)
dp.include_router(router)
dp.message.middleware(LoggingMiddleware())

async def main():
    print("Bot launched")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())