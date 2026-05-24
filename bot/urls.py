from django.urls import path

from .views import user_info
from .views import link_titles
from .views import index
from .views import get_icon


urlpatterns = [
    path("", index, name="index"),
    path("favicon.ico", get_icon, name="get_icon"),
    path("user_info", user_info, name="user_info"),
    path("link_titles", link_titles, name="link_titles")
]
