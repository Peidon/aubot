from django.urls import path

from .views import user_info
from .views import link_titles


urlpatterns = [
    path("user_info", user_info, name="user_info"),
    path("link_titles", link_titles, name="link_titles")
]
