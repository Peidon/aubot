import logging
import time
from logging.handlers import RotatingFileHandler

logging.basicConfig(level=logging.INFO, format='%(timestamp)s: %(duration)s: %(message)s')
handler = RotatingFileHandler("info.log", maxBytes=512000)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

# 1. Create a custom filter to provide 'timestamp'
class MetricsFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def filter(self, record):
        # Inject custom attributes into the record
        record.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        record.duration = f"{time.time() - self.start_time:.4f}s"
        return True

class RequestLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Code executed before the view (request phase)
        logger.addFilter(MetricsFilter())
        response = self.get_response(request)

        # Code executed after the view (response phase)

        if response.status_code == 201 and request.body:
            logger.info(f"{request.path} - {request.body} - {response.content}")

        return response