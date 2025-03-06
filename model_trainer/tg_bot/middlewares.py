from aiogram import BaseMiddleware
from aiogram.types import Message

class LoggingMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: Message, data: dict):
        print(f"Получено сообщение: {event.text}")
        if event.document:
            print(f"Получен файл: {event.document.file_name}, размер: {event.document.file_size} байт")
        return await handler(event, data)