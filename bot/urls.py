from django.urls import path

from .views import user_info
from .views import detect_fields


urlpatterns = [
    path("user_info", user_info, name="user_info"),
    path("detect_fields", detect_fields, name="detect_fields")
]
