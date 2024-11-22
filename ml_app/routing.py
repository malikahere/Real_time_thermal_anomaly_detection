from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/thermal_stream/$', consumers.ThermalStreamConsumer.as_asgi()),
]